"""The fixed candidate action set every arm chooses from.

Grid2Op's ``DiscreteActSpace`` enumerates one ``set_bus`` configuration per
substation per index, plus per-line connect/disconnect, plus do-nothing --
219 actions on case14. The enumeration is decoded here into flat arrays so
that (a) the compiled mask kernels can reason about it without touching
grid2op objects, and (b) the RDF backend can put every candidate action into
the graph, which is what makes masking a query.

The set is content-hashed. Every arm of the ablation must load the same hash;
two arms with different candidate sets are not comparable.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from enum import IntEnum
from typing import Any

import numpy as np

#: attr_to_keep for the discrete backbone. Mirrors ``actions.discrete_space``
#: in configs/base.yaml (frozen at step 2): pure ``set_bus``, 178 substation
#: configurations + do-nothing = 179 on case14. Line-status actions are still
#: understood by the decoder and the queries, they are just not in the frozen
#: set -- the reconnection heuristic handles reconnection for every arm.
DEFAULT_ACT_ATTRS: tuple[str, ...] = ("set_bus",)


class ActKind(IntEnum):
    NOTHING = 0
    SET_BUS = 1
    LINE_STATUS = 2


def _ro(arr, dtype) -> np.ndarray:
    out = np.array(arr, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False)
class CandidateActionSet:
    """Decoded, immutable view of a ``DiscreteActSpace``."""

    n: int
    attr_to_keep: tuple[str, ...]
    #: (n,) :class:`ActKind`
    kind: np.ndarray
    #: (n,) substation acted on for SET_BUS, else -1
    sub: np.ndarray
    #: (n,) line acted on for LINE_STATUS, else -1
    line: np.ndarray
    #: (n,) +1 reconnect / -1 disconnect for LINE_STATUS, else 0
    status: np.ndarray
    #: (n, dim_topo) target busbar per terminal for SET_BUS, 0 = untouched
    set_bus: np.ndarray
    fingerprint: str

    @classmethod
    def from_action_space(
        cls, action_space: Any, dim_topo: int, topo_sub: np.ndarray, attr_to_keep=DEFAULT_ACT_ATTRS
    ) -> CandidateActionSet:
        from grid2op.gym_compat import DiscreteActSpace

        das = DiscreteActSpace(action_space, attr_to_keep=list(attr_to_keep))
        n = int(das.n)
        kind = np.zeros(n, dtype=np.int8)
        sub = np.full(n, -1, dtype=np.int32)
        line = np.full(n, -1, dtype=np.int32)
        status = np.zeros(n, dtype=np.int8)
        set_bus = np.zeros((n, dim_topo), dtype=np.int8)

        for i in range(n):
            act = das.from_gym(i)
            sb = np.asarray(act.set_bus)
            ls = np.asarray(act.set_line_status)
            touched = np.flatnonzero(sb != 0)
            lines = np.flatnonzero(ls != 0)
            if touched.size and not lines.size:
                subs = np.unique(topo_sub[touched])
                if subs.size != 1:
                    raise ValueError(f"action {i} touches substations {subs.tolist()}; expected exactly one")
                if (sb[touched] < 1).any():
                    raise ValueError(f"action {i} assigns a terminal to busbar {sb[touched].min()}")
                kind[i] = ActKind.SET_BUS
                sub[i] = subs[0]
                set_bus[i] = sb
            elif lines.size == 1 and not touched.size:
                kind[i] = ActKind.LINE_STATUS
                line[i] = lines[0]
                status[i] = int(ls[lines[0]])
            elif not touched.size and not lines.size:
                kind[i] = ActKind.NOTHING
            else:
                raise ValueError(f"action {i} mixes set_bus and line status; expected one object per action")

        if int((kind == ActKind.NOTHING).sum()) != 1 or kind[0] != ActKind.NOTHING:
            raise ValueError("expected exactly one do-nothing action, at index 0")

        h = hashlib.sha256()
        for arr in (kind, sub, line, status, set_bus):
            h.update(np.ascontiguousarray(arr).tobytes())
        h.update(",".join(attr_to_keep).encode())
        return cls(
            n=n,
            attr_to_keep=tuple(attr_to_keep),
            kind=_ro(kind, np.int8),
            sub=_ro(sub, np.int32),
            line=_ro(line, np.int32),
            status=_ro(status, np.int8),
            set_bus=_ro(set_bus, np.int8),
            fingerprint=h.hexdigest()[:16],
        )

    # ------------------------------------------------------------------ #
    @property
    def is_nothing(self) -> np.ndarray:
        return self.kind == ActKind.NOTHING

    @property
    def is_set_bus(self) -> np.ndarray:
        return self.kind == ActKind.SET_BUS

    @property
    def is_line_status(self) -> np.ndarray:
        return self.kind == ActKind.LINE_STATUS

    def to_grid2op(self, index: int, action_space: Any):
        """Rebuild action ``index`` -- for tests and for the evaluation agent."""
        k = ActKind(int(self.kind[index]))
        if k == ActKind.NOTHING:
            return action_space()
        if k == ActKind.LINE_STATUS:
            return action_space({"set_line_status": [(int(self.line[index]), int(self.status[index]))]})
        terms = np.flatnonzero(self.set_bus[index])
        return action_space({"set_bus": [(int(t), int(self.set_bus[index, t])) for t in terms]})

    def describe(self, index: int, schema: Any = None) -> str:
        k = ActKind(int(self.kind[index]))
        if k == ActKind.NOTHING:
            return "do nothing"
        if k == ActKind.LINE_STATUS:
            verb = "reconnect" if self.status[index] > 0 else "disconnect"
            return f"{verb} line {self.line[index]}"
        terms = np.flatnonzero(self.set_bus[index])
        if schema is not None:
            parts = [f"{schema.terminal_label(int(t))}->{int(self.set_bus[index, t])}" for t in terms]
        else:
            parts = [f"t{int(t)}->{int(self.set_bus[index, t])}" for t in terms]
        return f"sub {self.sub[index]}: " + " ".join(parts)

    def __repr__(self) -> str:
        return (
            f"CandidateActionSet(n={self.n}, set_bus={int(self.is_set_bus.sum())}, "
            f"line_status={int(self.is_line_status.sum())}, fp={self.fingerprint})"
        )
