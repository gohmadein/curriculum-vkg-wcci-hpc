"""Static grid structure, extracted once per environment.

Everything here is invariant over an episode: which element hangs on which
substation, which topo_vect entry belongs to which element, which substations a
line joins. The dynamic half (what is connected to what *right now*) lives in
:mod:`grid2op_vkg.vkg.state`.

The central abstraction is the **terminal**: one entry of ``topo_vect``, i.e.
one attachment point that an action can move between busbars. Grid2Op's action
space is defined over exactly these, so the ontology is too.
"""

from __future__ import annotations

import dataclasses
import hashlib
from dataclasses import dataclass, field, fields
from enum import IntEnum
from typing import Any

import numpy as np

from grid2op_vkg.vkg.actions import DEFAULT_ACT_ATTRS, CandidateActionSet

# Column layout of ``grid_objects_types``; grid2op indexes its rows by
# topo_vect position, which is precisely what we want.
_SUB_COL, _LOAD_COL, _GEN_COL, _LOR_COL, _LEX_COL, _STORAGE_COL = range(6)


class ObjKind(IntEnum):
    """What kind of object a ``topo_vect`` entry belongs to."""

    LINE_OR = 0
    LINE_EX = 1
    GEN = 2
    LOAD = 3
    STORAGE = 4


#: ``grid_objects_types`` column that carries the object id, per kind.
_KIND_COL = {
    ObjKind.LINE_OR: _LOR_COL,
    ObjKind.LINE_EX: _LEX_COL,
    ObjKind.GEN: _GEN_COL,
    ObjKind.LOAD: _LOAD_COL,
    ObjKind.STORAGE: _STORAGE_COL,
}


def _get(obj: Any, name: str) -> Any:
    """Read a grid2op static attribute off an instance or its class."""
    if hasattr(obj, name):
        return getattr(obj, name)
    return getattr(type(obj), name)


def _frozen(arr: np.ndarray, dtype) -> np.ndarray:
    out = np.array(arr, dtype=dtype, copy=True)
    out.flags.writeable = False
    return out


@dataclass(frozen=True, eq=False)
class GridSchema:
    """Immutable static description of one Grid2Op grid.

    Build it with :meth:`from_env`. All arrays are read-only so a schema can be
    shared between the training env, the eval env and the RDF backend without
    anyone being able to corrupt it.
    """

    env_name: str

    n_sub: int
    n_line: int
    n_gen: int
    n_load: int
    n_storage: int
    dim_topo: int
    n_busbar_per_sub: int

    #: (n_sub,) number of terminals attached to each substation
    sub_info: np.ndarray
    #: (dim_topo,) substation id of each terminal
    topo_sub: np.ndarray
    #: (dim_topo,) :class:`ObjKind` of each terminal
    topo_kind: np.ndarray
    #: (dim_topo,) id of the object within its kind
    topo_obj: np.ndarray

    #: (n_line, 2) topo_vect positions of the origin and extremity terminals
    line_terminals: np.ndarray
    #: (n_line, 2) substation ids of the origin and extremity ends
    line_subs: np.ndarray
    #: (n_gen,) / (n_load,) / (n_storage,) topo_vect position of each unit
    gen_terminals: np.ndarray
    load_terminals: np.ndarray
    storage_terminals: np.ndarray

    #: (n_sub + 1,) CSR-style offsets into :attr:`sub_terminal_idx`
    sub_offsets: np.ndarray
    #: (dim_topo,) terminal ids grouped by substation
    sub_terminal_idx: np.ndarray

    #: (n_line,) thermal limit in amps, or all-NaN when the source had none
    thermal_limit: np.ndarray

    #: generator limits -- static, so a margin can be *derived* from the
    #: visible active power instead of read from a field the baseline lacks
    gen_pmax: np.ndarray
    gen_pmin: np.ndarray
    gen_max_ramp_up: np.ndarray
    gen_max_ramp_down: np.ndarray
    gen_redispatchable: np.ndarray

    name_sub: tuple[str, ...]
    name_line: tuple[str, ...]
    name_gen: tuple[str, ...]
    name_load: tuple[str, ...]
    name_storage: tuple[str, ...]

    #: The fixed candidate action set, when built from an object that has an
    #: ``action_space``. ``None`` for schemas built from an observation space.
    actions: CandidateActionSet | None = None

    #: The remedial-action catalogue (see :mod:`grid2op_vkg.vkg.remedial`):
    #: static electrical knowledge about what each candidate does to each
    #: line. ``None`` unless an agent loads one; not part of the fingerprint
    #: (the catalogue carries its own hash).
    remedial: Any | None = None

    _fingerprint: str = field(default="", compare=False)

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    @classmethod
    def from_env(
        cls, env: Any, env_name: str | None = None, act_attr_to_keep: tuple[str, ...] = DEFAULT_ACT_ATTRS
    ) -> GridSchema:
        """Extract the static structure from a grid2op environment.

        ``env`` may be an ``Environment``, an ``ObservationSpace`` or anything
        else deriving from grid2op's ``GridObjects``. When it carries an
        ``action_space`` the candidate action set is enumerated too.
        """
        n_sub = int(_get(env, "n_sub"))
        n_line = int(_get(env, "n_line"))
        n_gen = int(_get(env, "n_gen"))
        n_load = int(_get(env, "n_load"))
        n_storage = int(_get(env, "n_storage"))
        dim_topo = int(_get(env, "dim_topo"))

        got = _get(env, "grid_objects_types")
        if got.shape[0] != dim_topo:
            raise ValueError(f"grid_objects_types has {got.shape[0]} rows but dim_topo is {dim_topo}")

        topo_sub = got[:, _SUB_COL].astype(np.int32)
        topo_kind = np.full(dim_topo, -1, dtype=np.int8)
        topo_obj = np.full(dim_topo, -1, dtype=np.int32)
        for kind, col in _KIND_COL.items():
            sel = got[:, col] >= 0
            topo_kind[sel] = int(kind)
            topo_obj[sel] = got[sel, col]
        if (topo_kind < 0).any():
            bad = np.flatnonzero(topo_kind < 0)
            raise ValueError(f"topo_vect positions {bad.tolist()} match no object kind")

        line_terminals = np.stack(
            [_get(env, "line_or_pos_topo_vect"), _get(env, "line_ex_pos_topo_vect")], axis=1
        ).astype(np.int32)
        line_subs = np.stack([_get(env, "line_or_to_subid"), _get(env, "line_ex_to_subid")], axis=1).astype(np.int32)

        # terminals grouped by substation, in topo_vect order within a substation
        order = np.argsort(topo_sub, kind="stable").astype(np.int32)
        sub_offsets = np.zeros(n_sub + 1, dtype=np.int32)
        np.cumsum(np.bincount(topo_sub, minlength=n_sub), out=sub_offsets[1:])

        thermal = _get(env, "_thermal_limit_a") if hasattr(env, "_thermal_limit_a") else None
        if thermal is None:
            thermal = np.full(n_line, np.nan)

        actions = None
        if hasattr(env, "action_space") and act_attr_to_keep:
            actions = CandidateActionSet.from_action_space(env.action_space, dim_topo, topo_sub, act_attr_to_keep)

        schema = cls(
            env_name=env_name or str(_get(env, "env_name") if hasattr(env, "env_name") else "unknown"),
            n_sub=n_sub,
            n_line=n_line,
            n_gen=n_gen,
            n_load=n_load,
            n_storage=n_storage,
            dim_topo=dim_topo,
            n_busbar_per_sub=int(_get(env, "n_busbar_per_sub")),
            sub_info=_frozen(_get(env, "sub_info"), np.int32),
            topo_sub=_frozen(topo_sub, np.int32),
            topo_kind=_frozen(topo_kind, np.int8),
            topo_obj=_frozen(topo_obj, np.int32),
            line_terminals=_frozen(line_terminals, np.int32),
            line_subs=_frozen(line_subs, np.int32),
            gen_terminals=_frozen(_get(env, "gen_pos_topo_vect"), np.int32),
            load_terminals=_frozen(_get(env, "load_pos_topo_vect"), np.int32),
            storage_terminals=_frozen(_get(env, "storage_pos_topo_vect"), np.int32),
            sub_offsets=_frozen(sub_offsets, np.int32),
            sub_terminal_idx=_frozen(order, np.int32),
            thermal_limit=_frozen(thermal, np.float64),
            gen_pmax=_frozen(_get(env, "gen_pmax"), np.float64),
            gen_pmin=_frozen(_get(env, "gen_pmin"), np.float64),
            gen_max_ramp_up=_frozen(_get(env, "gen_max_ramp_up"), np.float64),
            gen_max_ramp_down=_frozen(_get(env, "gen_max_ramp_down"), np.float64),
            gen_redispatchable=_frozen(_get(env, "gen_redispatchable"), bool),
            name_sub=tuple(str(x) for x in _get(env, "name_sub")),
            name_line=tuple(str(x) for x in _get(env, "name_line")),
            name_gen=tuple(str(x) for x in _get(env, "name_gen")),
            name_load=tuple(str(x) for x in _get(env, "name_load")),
            name_storage=tuple(str(x) for x in _get(env, "name_storage")),
            actions=actions,
        )
        object.__setattr__(schema, "_fingerprint", schema._compute_fingerprint())
        schema.validate()
        return schema

    # ------------------------------------------------------------------ #
    # accessors
    # ------------------------------------------------------------------ #
    def terminals_of_sub(self, sub_id: int) -> np.ndarray:
        """Terminal (topo_vect) ids attached to ``sub_id``."""
        return self.sub_terminal_idx[self.sub_offsets[sub_id] : self.sub_offsets[sub_id + 1]]

    def sub_of_line(self, line_id: int) -> tuple[int, int]:
        return int(self.line_subs[line_id, 0]), int(self.line_subs[line_id, 1])

    @property
    def controllable_subs(self) -> np.ndarray:
        """Substations with enough terminals for a bus split to mean anything.

        With two terminals both non-trivial splits island one of them, so the
        candidate action set only ever touches substations with >= 3.
        """
        return np.flatnonzero(self.sub_info >= 3).astype(np.int32)

    @property
    def sub_line_adjacency(self) -> np.ndarray:
        """(n_sub, n_sub) bool: substations joined by at least one line.

        Purely structural -- it ignores line status and bus splitting. The
        *electrical* adjacency that reacts to the current topology is a query
        over :class:`~grid2op_vkg.vkg.state.GridState`.
        """
        adj = np.zeros((self.n_sub, self.n_sub), dtype=bool)
        o, e = self.line_subs[:, 0], self.line_subs[:, 1]
        adj[o, e] = True
        adj[e, o] = True
        np.fill_diagonal(adj, False)
        return adj

    def terminal_label(self, term_id: int) -> str:
        """Human-readable name, used for IRIs and for feature names."""
        kind = ObjKind(int(self.topo_kind[term_id]))
        obj = int(self.topo_obj[term_id])
        return {
            ObjKind.LINE_OR: f"line_{obj}_or",
            ObjKind.LINE_EX: f"line_{obj}_ex",
            ObjKind.GEN: f"gen_{obj}",
            ObjKind.LOAD: f"load_{obj}",
            ObjKind.STORAGE: f"storage_{obj}",
        }[kind]

    # ------------------------------------------------------------------ #
    # integrity
    # ------------------------------------------------------------------ #
    def validate(self) -> None:
        """Cross-check the redundant views of the same structure.

        Cheap, and it catches the kind of silent index mix-up that would
        otherwise show up as a mysteriously weak ablation arm.
        """
        if int(self.sub_info.sum()) != self.dim_topo:
            raise ValueError(f"sub_info sums to {self.sub_info.sum()}, expected dim_topo={self.dim_topo}")

        # every terminal is reachable exactly once through the substation index
        seen = np.concatenate([self.terminals_of_sub(s) for s in range(self.n_sub)])
        if not np.array_equal(np.sort(seen), np.arange(self.dim_topo)):
            raise ValueError("substation -> terminal index is not a partition of topo_vect")

        # line endpoints agree with the per-terminal substation map
        for end in (0, 1):
            got = self.topo_sub[self.line_terminals[:, end]]
            if not np.array_equal(got, self.line_subs[:, end]):
                raise ValueError(f"line_{'or' if end == 0 else 'ex'}_to_subid disagrees with grid_objects_types")

        for kind, terms in (
            (ObjKind.GEN, self.gen_terminals),
            (ObjKind.LOAD, self.load_terminals),
            (ObjKind.STORAGE, self.storage_terminals),
        ):
            if terms.size and not (self.topo_kind[terms] == int(kind)).all():
                raise ValueError(f"{kind.name} position vector points at terminals of another kind")
            if terms.size and not np.array_equal(self.topo_obj[terms], np.arange(terms.size)):
                raise ValueError(f"{kind.name} position vector is not in object order")

        if self.actions is not None:
            a = self.actions
            if a.set_bus.shape[1] != self.dim_topo:
                raise ValueError("candidate action set was built for a different dim_topo")
            for i in np.flatnonzero(a.is_set_bus):
                subs = np.unique(self.topo_sub[a.set_bus[i] != 0])
                if subs.size != 1 or subs[0] != a.sub[i]:
                    raise ValueError(f"action {i} claims substation {a.sub[i]} but touches {subs.tolist()}")

    def _compute_fingerprint(self) -> str:
        h = hashlib.sha256()
        h.update(self.env_name.encode())
        for f in fields(self):
            if f.name in ("_fingerprint", "remedial"):
                continue
            val = getattr(self, f.name)
            if isinstance(val, np.ndarray):
                h.update(np.ascontiguousarray(val).tobytes())
            elif isinstance(val, CandidateActionSet):
                h.update(val.fingerprint.encode())
            else:
                h.update(repr(val).encode())
        return h.hexdigest()[:16]

    @property
    def fingerprint(self) -> str:
        """Short content hash. Two arms must run against the same one."""
        return self._fingerprint

    def require_actions(self) -> CandidateActionSet:
        if self.actions is None:
            raise ValueError("schema has no candidate action set; build it from an env, not an observation space")
        return self.actions

    def with_remedial(self, catalogue: Any) -> GridSchema:
        """The same schema (same fingerprint) carrying a remedial-action catalogue."""
        a = self.require_actions()
        if catalogue.relief.shape != (a.n, self.n_line):
            raise ValueError(
                f"catalogue shape {catalogue.relief.shape} does not fit ({a.n} actions, {self.n_line} lines)"
            )
        return dataclasses.replace(self, remedial=catalogue)

    def __repr__(self) -> str:
        acts = f", {self.actions.n} actions" if self.actions is not None else ""
        return (
            f"GridSchema({self.env_name}, {self.n_sub} subs, {self.n_line} lines, "
            f"{self.n_gen} gens, {self.n_load} loads, {self.n_storage} storage, "
            f"dim_topo={self.dim_topo}{acts}, fp={self.fingerprint})"
        )
