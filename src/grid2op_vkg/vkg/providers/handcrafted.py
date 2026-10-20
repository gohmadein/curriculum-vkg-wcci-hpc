"""The Handcrafted arm: the same feedback, written as ordinary Python.

This file is the comparison the paper turns on, so its rules are stated once
more here in prose, and the implementation below follows them without going
through the ontology, the query layer, or the RDF backend. A static check in
:mod:`grid2op_vkg.vkg.fairness` refuses those imports; an equivalence test
asserts the outputs match :class:`~grid2op_vkg.vkg.providers.vkg.VkgProvider`
bit for bit on the same states and configuration.

Rules (v1, derived from what ExpertAgent does on case14 -- results/README.md):

  at risk      an in-service line with rho >= tau, or already on its overflow
               timer
  near risk    at risk, or within k hops -- over live-bus adjacency or, when
               configured, over substation adjacency (unchanged by splits)
  margin       max(tau_m - rho, 0) per in-service line
  pressure     sum of max(rho - tau, 0) over in-service lines
  headroom     upward redispatch room, min(pmax - p, ramp_up), of dispatchable
               generators on a bus touched by a near-risk line
  split        a substation whose live terminals sit on more than one bus
  controllable a substation off cooldown with >= 3 terminals

  relevant     do nothing; reconnect anything; if nothing is at risk, merge a
               split substation back; if something is, reconfigure an endpoint
               substation of a near-risk line in a way that changes it, or drop
               a line sharing a bus with the overloaded one
  islanding    a bus split that uses a busbar with fewer than two terminals
               or with no line on it (a lone line end is an implicit disconnection)

Everything the VKG arm knows, this arm knows too. What the VKG arm has and this
arm lacks is a place to put the knowledge that is not this file.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from grid2op_vkg.vkg.actions import ActKind
from grid2op_vkg.vkg.config import VkgConfig
from grid2op_vkg.vkg.providers.base import FeedbackProvider, SearchGuidance
from grid2op_vkg.vkg.schema import GridSchema

#: Same declared-box ceiling as the engine's; see the note there.
_BOX_BOUND = 1.0e3


class HandcraftedProvider(FeedbackProvider):
    name = "handcrafted"
    required_obs_attrs = frozenset(
        {
            "topo_vect",
            "line_status",
            "rho",
            "timestep_overflow",
            "time_before_cooldown_line",
            "time_before_cooldown_sub",
            "gen_p",
        }
    )

    def __init__(self, schema: GridSchema, cfg: VkgConfig) -> None:
        self.schema = schema
        self.cfg = cfg
        self.tau = float(cfg.risk_tau)
        self.k = int(cfg.relevance_k)
        self.adjacency = str(cfg.relevance_adjacency)
        self.margin_tau = float(cfg.margin_tau)
        s = schema
        # static helpers
        self._line_term_mask = np.zeros(s.dim_topo, dtype=bool)
        self._line_term_mask[s.line_terminals.ravel()] = True
        self._line_sub = np.zeros((s.n_line, s.n_sub), dtype=bool)
        self._line_sub[np.arange(s.n_line), s.line_subs[:, 0]] = True
        self._line_sub[np.arange(s.n_line), s.line_subs[:, 1]] = True
        # lines meeting at a substation, whatever its configuration (static)
        self._sub_adj = np.zeros((s.n_line, s.n_line), dtype=bool)
        for ea in (0, 1):
            for eb in (0, 1):
                self._sub_adj |= s.line_subs[:, ea][:, None] == s.line_subs[:, eb][None, :]
        np.fill_diagonal(self._sub_adj, False)
        self._islanding = self._static_islanding() if s.actions is not None else None
        self._obs: Any = None
        self._cache: dict[str, Any] = {}
        self._t0 = 0.0
        self._ms: list[float] = []

    # ------------------------------------------------------------------ #
    def begin_step(self, obs: Any) -> None:
        import time

        self._t0 = time.perf_counter()
        self._obs = obs
        self._cache = {}

    def end_step(self) -> float:
        import time

        ms = (time.perf_counter() - self._t0) * 1e3
        self._ms.append(ms)
        return ms

    # -- primitive views ----------------------------------------------- #
    def _memo(self, key, fn):
        if key not in self._cache:
            self._cache[key] = fn()
        return self._cache[key]

    @property
    def _bus(self) -> np.ndarray:
        """Global bus id per terminal; -1 if off or on a dead line."""

        def f():
            s, o = self.schema, self._obs
            tv = np.asarray(o.topo_vect)
            bus = np.where(tv > 0, s.topo_sub + (tv - 1) * s.n_sub, -1).astype(np.int32)
            dead = s.line_terminals[~np.asarray(o.line_status, dtype=bool)]
            bus[dead.ravel()] = -1
            return bus

        return self._memo("bus", f)

    @property
    def _adjacent(self) -> np.ndarray:
        def f():
            lb = self._bus[self.schema.line_terminals]  # (n_line, 2)
            n = self.schema.n_line
            adj = np.zeros((n, n), dtype=bool)
            for i in range(n):
                for end in (0, 1):
                    b = lb[i, end]
                    if b >= 0:
                        adj[i] |= (lb == b).any(axis=1)
            np.fill_diagonal(adj, False)
            return adj

        return self._memo("adj", f)

    @property
    def _at_risk(self) -> np.ndarray:
        def f():
            o = self._obs
            live = np.asarray(o.line_status, dtype=bool)
            return live & ((np.asarray(o.rho) >= self.tau) | (np.asarray(o.timestep_overflow) > 0))

        return self._memo("risk", f)

    @property
    def _near_risk(self) -> np.ndarray:
        def f():
            adj = self._sub_adj if self.adjacency == "substation" else self._adjacent
            reach = self._at_risk.copy()
            frontier = reach.copy()
            for _ in range(self.k):
                if not frontier.any():
                    break
                nxt = adj[frontier].any(axis=0) & ~reach
                reach |= nxt
                frontier = nxt
            return reach

        return self._memo("near", f)

    @property
    def _split(self) -> np.ndarray:
        def f():
            s, bus = self.schema, self._bus
            out = np.zeros(s.n_sub, dtype=bool)
            for sub in range(s.n_sub):
                b = bus[s.terminals_of_sub(sub)]
                b = b[b >= 0]
                out[sub] = b.size > 1 and (b != b[0]).any()
            return out

        return self._memo("split", f)

    # -- feature blocks -------------------------------------------------- #
    def _block(self, name: str) -> np.ndarray:
        o, s = self._obs, self.schema
        live = np.asarray(o.line_status, dtype=bool)
        rho = np.asarray(o.rho, dtype=np.float64)
        if name == "lines_at_risk":
            return self._at_risk
        if name == "lines_near_risk":
            return self._near_risk
        if name == "line_margin":
            return np.where(live, np.maximum(self.margin_tau - rho, 0.0), 0.0)
        if name == "overload_pressure":
            return np.array(np.where(live, np.maximum(rho - self.tau, 0.0), 0.0).sum())
        if name == "headroom_near_risk":
            near = self._near_risk
            if not near.any():
                return np.array(0.0)
            buses = self._bus[s.line_terminals[near]].ravel()
            buses = buses[buses >= 0]
            gen_bus = self._bus[s.gen_terminals]
            close = (gen_bus >= 0) & np.isin(gen_bus, buses)
            # upward room = min(pmax - p, ramp up), nothing for non-dispatchable units
            up = np.minimum(s.gen_pmax - np.asarray(o.gen_p, dtype=np.float64), s.gen_max_ramp_up)
            up = np.where(s.gen_redispatchable, np.maximum(up, 0.0), 0.0)
            return np.array(up[close].sum())
        if name == "sub_is_split":
            return self._split
        if name == "controllable_subs":
            return (np.asarray(o.time_before_cooldown_sub) == 0) & (s.sub_info >= 3)
        raise KeyError(f"handcrafted arm has no feature block named {name!r}")

    _BLOCK_INFO = {
        "lines_at_risk": ("line", 0.0, 1.0),
        "lines_near_risk": ("line", 0.0, 1.0),
        "line_margin": ("line", 0.0, 2.0),
        "overload_pressure": ("scalar", 0.0, np.inf),
        "headroom_near_risk": ("scalar", 0.0, np.inf),
        "sub_is_split": ("sub", 0.0, 1.0),
        "controllable_subs": ("sub", 0.0, 1.0),
    }

    def _size(self, kind: str) -> int:
        return {"line": self.schema.n_line, "sub": self.schema.n_sub, "scalar": 1}[kind]

    def features(self) -> np.ndarray | None:
        if not self.cfg.state:
            return None
        parts = [np.asarray(self._block(n), dtype=np.float32).reshape(-1) for n in self.cfg.state_queries]
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    @property
    def feature_names(self) -> list[str]:
        if not self.cfg.state:
            return []
        names = []
        for n in self.cfg.state_queries:
            kind = self._BLOCK_INFO[n][0]
            if kind == "scalar":
                names.append(n)
            else:
                labels = self.schema.name_line if kind == "line" else self.schema.name_sub
                names.extend(f"{n}[{lab}]" for lab in labels)
        return names

    def feature_bounds(self):
        low, high = [], []
        for n in self.cfg.state_queries if self.cfg.state else ():
            kind, lo, hi = self._BLOCK_INFO[n]
            size = self._size(kind)
            low.extend([float(np.clip(lo, -_BOX_BOUND, _BOX_BOUND))] * size)
            high.extend([float(np.clip(hi, -_BOX_BOUND, _BOX_BOUND))] * size)
        return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        if not self.cfg.state:
            return 0
        return sum(self._size(self._BLOCK_INFO[n][0]) for n in self.cfg.state_queries)

    # -- reward ---------------------------------------------------------- #
    def potential(self) -> float | None:
        if not self.cfg.reward:
            return None
        return -float(self._block("overload_pressure"))

    # -- mask ------------------------------------------------------------ #
    def _static_islanding(self) -> np.ndarray:
        a = self.schema.require_actions()
        out = np.zeros(a.n, dtype=bool)
        for i in np.flatnonzero(a.is_set_bus):
            cfg = a.set_bus[i]
            for b in (1, 2):
                on_b = cfg == b
                # a used busbar needs two terminals and a line: a lone terminal is a
                # dangling line end (an implicit disconnection), injections alone island
                if on_b.any() and (on_b.sum() < 2 or not (on_b & self._line_term_mask).any()):
                    out[i] = True
        return out

    def _relevant(self) -> np.ndarray:
        a, o = self.schema.require_actions(), self._obs
        out = a.kind == ActKind.NOTHING
        out |= (a.kind == ActKind.LINE_STATUS) & (a.status == 1)
        risk = self._at_risk
        sb = a.kind == ActKind.SET_BUS
        if not risk.any():
            merge_back = sb & ~(a.set_bus == 2).any(axis=1)
            out |= merge_back & self._split[np.maximum(a.sub, 0)]
            return out
        near_subs = self._line_sub[self._near_risk].any(axis=0)
        live_term = self._bus >= 0
        cur = np.where(live_term, np.asarray(o.topo_vect), -999)
        changes = ((a.set_bus != 0) & live_term[None, :] & (a.set_bus != cur[None, :])).any(axis=1)
        out |= sb & near_subs[np.maximum(a.sub, 0)] & changes
        adj_to_risk = self._adjacent[risk].any(axis=0) & ~risk
        out |= (a.kind == ActKind.LINE_STATUS) & (a.status == -1) & adj_to_risk[np.maximum(a.line, 0)]
        if self.cfg.remedial_membership:
            # electrical relevance: the catalogue says the action lowers a risk line's loading
            out |= sb & changes & (self._remedial_score() >= float(self.cfg.remedial_theta))
        return out

    # -- knowledge layer (twin of queries/remedial.py) ------------------- #
    def _remedial_score(self) -> np.ndarray:
        """Per action, max(0, predicted reduction of the max loading); zeros without a catalogue or risk."""
        a, cat, risk = self.schema.require_actions(), self.schema.remedial, self._at_risk
        if cat is None or not risk.any():
            return np.zeros(a.n, dtype=np.float64)
        if "remedial_score" not in self._cache:
            o = self._obs
            self._cache["remedial_score"] = cat.reduction_score(
                np.asarray(o.rho, dtype=np.float64), np.asarray(o.line_status, dtype=bool)
            )
        return self._cache["remedial_score"]

    def action_mask(self, action_set: Any = None) -> np.ndarray | None:
        """Semantic filter only; legality is shared plumbing (channels.mask)."""
        if not self.cfg.mask:
            return None
        return self._relevant() & ~self._islanding

    # -- provenance ------------------------------------------------------ #
    def justify(self, action_idx: int) -> np.ndarray:
        """Hand-written twin of ExplainAction: (risk line, substation) witnesses for one action."""
        a, s = self.schema.require_actions(), self.schema
        out = np.zeros((s.n_line, s.n_sub), dtype=bool)
        if a.kind[action_idx] != ActKind.SET_BUS:
            return out
        sub = int(a.sub[action_idx])
        risk = self._at_risk
        if not risk.any():
            return out
        adj = self._sub_adj if self.adjacency == "substation" else self._adjacent
        for r in np.flatnonzero(risk):
            reach = np.zeros(s.n_line, dtype=bool)
            reach[r] = True
            frontier = reach.copy()
            for _ in range(self.k):
                if not frontier.any():
                    break
                nxt = adj[frontier].any(axis=0) & ~reach
                reach |= nxt
                frontier = nxt
            if self._line_sub[reach, sub].any():
                out[r, sub] = True
        return out

    # -- budget layer ---------------------------------------------------- #
    def search_guidance(self) -> SearchGuidance:
        """The hand-written twin of the four budget queries.

        Same answers, written straight against the observation. Asserted
        element-for-element against the VKG side in
        ``tests/test_vkg_handcrafted.py``.
        """
        a, o, s = self.schema.require_actions(), self._obs, self.schema
        sb = a.kind == ActKind.SET_BUS
        sub_of = np.maximum(a.sub, 0)
        split = self._split

        live_term = self._bus >= 0
        moved = (a.set_bus != 0) & live_term[None, :] & (a.set_bus != np.asarray(o.topo_vect)[None, :])
        disruption = moved.sum(axis=1).astype(np.float64)

        rho = np.asarray(o.rho)
        priority_subs = np.zeros(s.n_sub, dtype=bool)
        if rho.size:
            priority_subs[s.line_subs[rho >= rho.max()].ravel()] = True

        opens = sb & (a.set_bus == 2).any(axis=1)
        cap = int(self.cfg.max_split_subs)
        if int(split.sum()) < cap:
            affordable = np.ones(a.n, dtype=bool)
        else:
            affordable = ~(opens & ~split[sub_of])

        merge_back = sb & ~(a.set_bus == 2).any(axis=1)
        cd = np.asarray(o.time_before_cooldown_sub)
        recovery = merge_back & (split & (cd == 0))[sub_of]

        return SearchGuidance(
            disruption=disruption,
            priority_subs=priority_subs,
            affordable=affordable,
            recovery=recovery,
            split=split,
            reconnectable=(~np.asarray(o.line_status, dtype=bool))
            & (np.asarray(o.time_before_cooldown_line) == 0),
            at_risk=self._at_risk,
            remedial=self._remedial_score() >= float(self.cfg.remedial_theta),
            remedial_score=self._remedial_score(),
        )

    # -- aux ------------------------------------------------------------- #
    def aux_targets(self) -> dict[str, np.ndarray] | None:
        if not self.cfg.aux:
            return None
        return {
            "lines_near_risk": np.asarray(self._near_risk, dtype=np.float32),
            "sub_is_split": np.asarray(self._split, dtype=np.float32),
        }

    def timings(self) -> dict[str, float]:
        if not self._ms:
            return {"n_steps": 0}
        xs = np.asarray(self._ms)
        return {
            "vkg_step_ms_p50": float(np.percentile(xs, 50)),
            "vkg_step_ms_p95": float(np.percentile(xs, 95)),
            "n_steps": len(self._ms),
            "backend": "handcrafted",
        }
