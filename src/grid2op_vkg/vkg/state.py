"""One step of dynamic grid state.

A :class:`GridState` is a snapshot of the handful of observation fields the VKG
queries are allowed to read, plus the derived bus assignment. It is deliberately
a *copy*: the reference RDF backend and the compiled kernel must see byte
identical inputs, and grid2op is free to recycle observation buffers.

Only the fields listed in :data:`REQUIRED_OBS_ATTRS` are ever read. That set is
what the fairness check compares against the Raw arm's observation list -- the
VKG side must not see anything the baseline cannot.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from grid2op_vkg.vkg.schema import GridSchema

#: Observation attributes the VKG state layer reads. Keep this honest: adding a
#: name here widens what every VKG arm can see and must be mirrored in the Raw
#: arm's ``obs_attr_to_keep`` or the comparison is not fair.
REQUIRED_OBS_ATTRS: frozenset[str] = frozenset(
    {
        "topo_vect",
        "line_status",
        "rho",
        "timestep_overflow",
        "time_before_cooldown_line",
        "time_before_cooldown_sub",
        "gen_p",
        "actual_dispatch",
    }
)

#: Bus id meaning "this terminal is disconnected".
DISCONNECTED = -1


def _ro(arr: Any, dtype) -> np.ndarray:
    out = np.array(arr, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


@dataclass(eq=False)
class GridState:
    """Immutable-by-convention snapshot of the dynamic grid state."""

    schema: GridSchema
    step: int

    topo_vect: np.ndarray  # (dim_topo,) local busbar 1..n_busbar, or -1
    line_status: np.ndarray  # (n_line,) bool
    rho: np.ndarray  # (n_line,)
    timestep_overflow: np.ndarray  # (n_line,)
    cooldown_line: np.ndarray  # (n_line,)
    cooldown_sub: np.ndarray  # (n_sub,)
    gen_p: np.ndarray  # (n_gen,)
    actual_dispatch: np.ndarray  # (n_gen,)

    def __post_init__(self) -> None:
        self._cache: dict[str, Any] = {}

    # ------------------------------------------------------------------ #
    @classmethod
    def from_obs(cls, schema: GridSchema, obs: Any, step: int = -1) -> GridState:
        """Snapshot a grid2op observation.

        ``step`` defaults to the observation's own ``current_step`` when it has
        one, so the RDF backend can stamp triples with a valid time.
        """
        if step < 0:
            step = int(getattr(obs, "current_step", -1))
        return cls(
            schema=schema,
            step=step,
            topo_vect=_ro(obs.topo_vect, np.int32),
            line_status=_ro(obs.line_status, bool),
            rho=_ro(obs.rho, np.float64),
            timestep_overflow=_ro(obs.timestep_overflow, np.int32),
            cooldown_line=_ro(obs.time_before_cooldown_line, np.int32),
            cooldown_sub=_ro(obs.time_before_cooldown_sub, np.int32),
            gen_p=_ro(obs.gen_p, np.float64),
            actual_dispatch=_ro(obs.actual_dispatch, np.float64),
        )

    # ------------------------------------------------------------------ #
    # derived views, computed at most once per step
    # ------------------------------------------------------------------ #
    def _memo(self, key: str, fn) -> Any:
        if key not in self._cache:
            val = fn()
            if isinstance(val, np.ndarray):
                val.flags.writeable = False
            self._cache[key] = val
        return self._cache[key]

    @property
    def gen_margin_up(self) -> np.ndarray:
        """Upward redispatch room, derived: ``min(pmax - p, ramp_up)``, 0 if not dispatchable.

        Grid2Op exposes this as ``obs.gen_margin_up``, but the frozen baseline
        observation does not include that field -- and it does not need to: it
        is a function of the visible ``gen_p`` and static limits. Deriving it
        here is exactly the "same information, better representation" the VKG
        arm is supposed to deliver. A test checks it equals grid2op's value.
        """

        def build() -> np.ndarray:
            s = self.schema
            up = np.minimum(s.gen_pmax - self.gen_p, s.gen_max_ramp_up)
            return np.where(s.gen_redispatchable, np.maximum(up, 0.0), 0.0)

        return self._memo("gen_margin_up", build)

    @property
    def gen_margin_down(self) -> np.ndarray:
        def build() -> np.ndarray:
            s = self.schema
            dn = np.minimum(self.gen_p - s.gen_pmin, s.gen_max_ramp_down)
            return np.where(s.gen_redispatchable, np.maximum(dn, 0.0), 0.0)

        return self._memo("gen_margin_down", build)

    @property
    def global_bus(self) -> np.ndarray:
        """(dim_topo,) global electrical node id per terminal, -1 if disconnected.

        Grid2Op numbers busbar ``b`` of substation ``s`` as ``s + (b-1)*n_sub``;
        two terminals share an electrical node exactly when this value matches.
        """

        def build() -> np.ndarray:
            s = self.schema
            gb = s.topo_sub.astype(np.int32) + (self.topo_vect - 1) * s.n_sub
            return np.where(self.topo_vect > 0, gb, DISCONNECTED).astype(np.int32)

        return self._memo("global_bus", build)

    @property
    def terminal_bus(self) -> np.ndarray:
        """(dim_topo,) bus each terminal is *electrically* on, -1 otherwise.

        Same as :attr:`global_bus` except that both ends of an out-of-service
        line are forced off: such a line carries no flow whatever its topo_vect
        entries say. This is the single source of truth for connectivity -- the
        RDF backend emits ``g2o:connectedTo`` from exactly this array.
        """

        def build() -> np.ndarray:
            tb = self.global_bus.copy()
            dead = self.schema.line_terminals[~self.line_status]
            tb[dead.ravel()] = DISCONNECTED
            return tb.astype(np.int32)

        return self._memo("terminal_bus", build)

    @property
    def line_bus(self) -> np.ndarray:
        """(n_line, 2) global bus of each line end, -1 when the end is off."""
        return self._memo("line_bus", lambda: self.terminal_bus[self.schema.line_terminals].astype(np.int32))

    @property
    def active_buses(self) -> np.ndarray:
        """Sorted ids of the electrical nodes that currently have a terminal."""
        return self._memo("active_buses", lambda: np.unique(self.global_bus[self.global_bus >= 0]).astype(np.int32))

    @property
    def sub_is_split(self) -> np.ndarray:
        """(n_sub,) bool: substation currently spread over more than one Bus.

        Defined on :attr:`terminal_bus`, not on ``topo_vect``: a terminal of an
        out-of-service line sits on no electrical node at all, so it cannot make
        its substation count as split.
        """

        def build() -> np.ndarray:
            s = self.schema
            tb = self.terminal_bus
            out = np.zeros(s.n_sub, dtype=bool)
            for sub in range(s.n_sub):
                live = tb[s.terminals_of_sub(sub)]
                live = live[live >= 0]
                out[sub] = live.size > 1 and bool((live != live[0]).any())
            return out

        return self._memo("sub_is_split", build)

    @property
    def line_adjacency(self) -> np.ndarray:
        """(n_line, n_line) bool: in-service lines sharing an electrical node.

        This is the adjacency that actually matters for flow transfer, and it
        changes whenever a substation splits -- which is exactly the structure
        the raw observation vector leaves implicit.
        """

        def build() -> np.ndarray:
            lb = self.line_bus  # (n_line, 2)
            n = self.schema.n_line
            adj = np.zeros((n, n), dtype=bool)
            for end_a in (0, 1):
                for end_b in (0, 1):
                    a = lb[:, end_a][:, None]
                    b = lb[:, end_b][None, :]
                    adj |= (a >= 0) & (a == b)
            np.fill_diagonal(adj, False)
            return adj

        return self._memo("line_adjacency", build)

    def k_hop_lines(self, sources: np.ndarray, k: int) -> np.ndarray:
        """(n_line,) bool mask of lines within ``k`` electrical hops of ``sources``.

        ``k=0`` returns the sources themselves.
        """
        reach = np.zeros(self.schema.n_line, dtype=bool)
        reach[np.asarray(sources, dtype=np.int64)] = True
        if k <= 0:
            return reach
        adj = self.line_adjacency
        frontier = reach.copy()
        for _ in range(k):
            nxt = adj[frontier].any(axis=0) & ~reach
            if not nxt.any():
                break
            reach |= nxt
            frontier = nxt
        return reach

    def __repr__(self) -> str:
        return (
            f"GridState(step={self.step}, max_rho={self.rho.max():.3f}, "
            f"n_disconnected={int((~self.line_status).sum())}, "
            f"n_split={int(self.sub_is_split.sum())})"
        )
