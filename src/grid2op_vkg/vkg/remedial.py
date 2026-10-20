"""The remedial-action catalogue: what each candidate action does to each line's loading.

Static electrical knowledge, compiled offline with the same simulator the agents
use: for K snapshots of the grid in its reference topology (train chronics,
day-1 steps), every ``set_bus`` candidate is simulated on the *current*
injections and

    effect[a, l] = rho_l(do nothing) - rho_l(a)

is averaged over the snapshots -- **signed**: positive lowers line ``l``'s
loading, negative raises it. Pairs with |mean effect| below ``keep_min`` are
dropped. It is the remedial-action-scheme table of a control room, and the
analogue of OptimCVXPY building its DC model from the grid file: once per
grid, outside any episode, no test or validation chronic involved.

In the graph it is a reified relation (``g2o:RemedialEffect`` with
``g2o:ofAction`` / ``g2o:onLine`` / ``g2o:effect``) joined at query time with
the live loading of every line, which gives a *predicted post-action loading
profile* per candidate:

    predicted_rho[a, l] = rho_l - effect[a, l]

The offline oracle on the validation deaths showed why the sign matters: the
relief on the stressed line alone ranks candidates no better than a uniform
draw (a split that unloads one corridor loads another), while the predicted
*maximum* loading -- side effects included -- correlates with the simulated
outcome (Spearman ~0.47) and moves the best candidate from median rank ~900
to ~120. Membership and ordering therefore both use the predicted max.

The compiled kernels read the dense matrix; the SPARQL path reads the same
pairs as triples (RDF) or through the ``remedial_effect`` table (Ontop, every
schema shift). ``GridSchema.remedial`` is ``None`` unless an agent loads a
catalogue, and then nothing here is consulted.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


def _frozen(arr: np.ndarray) -> np.ndarray:
    arr = np.ascontiguousarray(arr, dtype=np.float32)
    arr.flags.writeable = False
    return arr


@dataclass(frozen=True, eq=False)
class RemedialCatalogue:
    #: (n_action, n_line) mean signed effect rho(nothing) - rho(action); 0 where absent (|effect| < keep_min)
    relief: np.ndarray
    #: (n_action, n_line) fraction of snapshots with effect > 0.01 (sign stability); 0 where absent
    frac_pos: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)
    # sparse view of the effects, built once: CSR-style (indptr, indices, data) and a
    # boolean "has an effect" matrix, so the per-step prediction is O(nnz + n_action*n_line bytes)
    _indptr: np.ndarray = field(init=False, repr=False, compare=False)
    _indices: np.ndarray = field(init=False, repr=False, compare=False)
    _data: np.ndarray = field(init=False, repr=False, compare=False)
    _has: np.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        rows, cols = np.nonzero(self.relief)  # row-major order == sorted by action
        counts = np.bincount(rows, minlength=self.relief.shape[0])
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        has = self.relief != 0
        has.flags.writeable = False
        object.__setattr__(self, "_indptr", indptr)
        object.__setattr__(self, "_indices", cols.astype(np.int64))
        object.__setattr__(self, "_data", self.relief[rows, cols].astype(np.float64))
        object.__setattr__(self, "_has", has)

    @property
    def n_pairs(self) -> int:
        return int((self.relief != 0).sum())

    def pairs(self, dense: bool = False) -> np.ndarray:
        """(k, 2) int array of (action, line) pairs.

        Sparse (default): pairs with a catalogued effect of either sign -- what
        the Parquet file stores. Dense: every (action, line) pair, zeros
        included -- what the graph paths (RDF, relational) emit, so that the
        ``predicted_max_rho`` query is a plain join with no OPTIONAL / NOT
        EXISTS: a stored zero and an absent pair mean the same thing ("no
        measurable effect") and the total relation keeps SPARQL evaluation
        linear in the table.
        """
        if dense:
            n_a, n_l = self.relief.shape
            return np.column_stack([np.repeat(np.arange(n_a), n_l), np.tile(np.arange(n_l), n_a)])
        return np.argwhere(self.relief != 0)

    def predicted_max_rho(self, rho: np.ndarray, in_service: np.ndarray | None = None) -> np.ndarray:
        """Per action, the predicted max loading after it: max over in-service lines of rho_l - effect[a, l].

        Lines without a catalogued effect keep their current loading; actions
        without any catalogued effect (do-nothing, line status) predict the
        current maximum. This is the compiled twin of the ``predicted_max_rho``
        query.
        """
        rho = np.asarray(rho, dtype=np.float64)
        live = np.ones(rho.size, dtype=bool) if in_service is None else np.asarray(in_service, dtype=bool)
        n_a = self.relief.shape[0]
        if not live.any():
            return np.full(n_a, -np.inf)
        r = np.where(live, rho, -np.inf)
        # lines the action does not touch keep their loading: the hottest live line
        # *outside* the action's catalogued set, found by scanning lines in
        # descending loading until one the action has no effect on
        order = np.argsort(-r, kind="stable")
        has_sorted = self._has[:, order]
        first = np.argmax(~has_sorted, axis=1)
        uncatalogued = r[order][first]
        uncatalogued[has_sorted.all(axis=1)] = -np.inf
        # lines the action touches: rho_l - effect, segment max per action (float64, exact)
        vals = r[self._indices] - self._data
        seg = np.full(n_a, -np.inf)
        nz = self._indptr[1:] > self._indptr[:-1]
        if nz.any():
            seg[nz] = np.maximum.reduceat(vals, self._indptr[:-1][nz])
        return np.maximum(uncatalogued, seg)

    # ------------------------------------------------------------------ #
    @classmethod
    def from_matrix(
        cls,
        relief: np.ndarray,
        frac_pos: np.ndarray | None = None,
        meta: dict[str, Any] | None = None,
        keep_min: float = 0.01,
    ) -> RemedialCatalogue:
        relief = np.asarray(relief, dtype=np.float64)
        keep = np.isfinite(relief) & (np.abs(relief) >= keep_min)
        fp = np.ones_like(relief) if frac_pos is None else np.asarray(frac_pos, dtype=np.float64)
        return cls(
            _frozen(np.where(keep, relief, 0.0)),
            _frozen(np.where(keep, np.nan_to_num(fp), 0.0)),
            dict(meta or {}, keep_min=float(keep_min)),
        )

    @classmethod
    def compile(
        cls, env: Any, schema: Any, observations: list[Any], keep_min: float = 0.01, min_pos: float = 0.01
    ) -> RemedialCatalogue:
        """Compile from observations in the reference topology (the per-step simulate cap must be lifted).

        Used by the tests on case14 (one snapshot, 178 actions, well under a
        second) and, snapshot by snapshot, by ``scripts/build_catalogue.py``.
        """
        acts = schema.require_actions()
        R = np.full((len(observations), acts.n, schema.n_line), np.nan, dtype=np.float32)
        for k, obs in enumerate(observations):
            base, _, done, info = obs.simulate(env.action_space(), time_step=0)
            if done or info.get("exception"):
                raise ValueError("doing nothing must simulate cleanly at a compilation snapshot")
            base_rho = np.asarray(base.rho, dtype=np.float64)
            for i in np.flatnonzero(acts.is_set_bus):
                try:
                    so, _, d, inf = obs.simulate(acts.to_grid2op(int(i), env.action_space), time_step=0)
                except Exception:  # noqa: BLE001  an action the simulator rejects has no catalogued effect
                    continue
                if d or inf.get("exception") or inf.get("is_illegal") or inf.get("is_ambiguous"):
                    continue
                R[k, i] = base_rho - np.asarray(so.rho, dtype=np.float64)
        with np.errstate(all="ignore"):
            mean = np.nanmean(R, axis=0)
            frac = np.nanmean(R > min_pos, axis=0)
        return cls.from_matrix(mean, frac, {"n_snapshots": len(observations), "source": "compile"}, keep_min)

    def reduction_score(self, rho: np.ndarray, in_service: np.ndarray | None = None) -> np.ndarray:
        """Per action, max(0, max live rho - predicted_max_rho), exact, at a fraction of the cost.

        The predicted maximum is at least ``rho_l - effect[a, l]`` for every
        line ``l`` at the current maximum, so an action can only lower the
        maximum if it relieves *every* such line. The full prediction is
        computed for those rows only -- a few hundred to a few thousand on
        wcci_2022 instead of 72 958 -- and every other action scores exactly 0:
        its true reduction is <= 0 and the score is clipped there. Same float64
        arithmetic as :meth:`predicted_max_rho`, so the rows it computes are
        bit-identical to the full computation.
        """
        rho = np.asarray(rho, dtype=np.float64)
        live = np.ones(rho.size, dtype=bool) if in_service is None else np.asarray(in_service, dtype=bool)
        out = np.zeros(self.relief.shape[0], dtype=np.float64)
        if not live.any():
            return out
        r = np.where(live, rho, -np.inf)
        now = float(r.max())
        top = np.flatnonzero(r == now)
        rows = np.flatnonzero((self.relief[:, top] > 0).all(axis=1))
        if rows.size == 0:
            return out
        pred = (rho[live][None, :] - self.relief[rows][:, live].astype(np.float64)).max(axis=1)
        out[rows] = np.maximum(now - pred, 0.0)
        return out

    # ------------------------------------------------------------------ #
    @staticmethod
    def default_path(env_name: str, actions_fingerprint: str, root: str | Path = "data/catalogue") -> Path:
        return Path(root) / f"{env_name}_{actions_fingerprint}.parquet"

    def save(self, path: str | Path) -> Path:
        """Parquet + JSON side-car. Written through DuckDB, like the relational layer (no pyarrow needed)."""
        import duckdb
        import pandas as pd

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        pairs = self.pairs()
        df = pd.DataFrame(
            {
                "action": pairs[:, 0].astype(np.int32),
                "line": pairs[:, 1].astype(np.int32),
                "relief": self.relief[pairs[:, 0], pairs[:, 1]].astype(np.float32),
                "frac_pos": self.frac_pos[pairs[:, 0], pairs[:, 1]].astype(np.float32),
            }
        )
        con = duckdb.connect()
        try:
            con.register("cat_df", df)
            con.execute(f"COPY cat_df TO '{path.resolve()}' (FORMAT PARQUET)")
        finally:
            con.close()
        path.with_suffix(".json").write_text(json.dumps(dict(self.meta, n_pairs=int(len(pairs))), indent=1))
        return path

    @classmethod
    def load(cls, path: str | Path, n_action: int, n_line: int) -> RemedialCatalogue:
        import duckdb

        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"no remedial catalogue at {path}; build one with scripts/build_catalogue.py")
        con = duckdb.connect()
        try:
            rows = con.execute(
                f"SELECT action, line, relief, frac_pos FROM read_parquet('{path.resolve()}')"
            ).fetchnumpy()
        finally:
            con.close()
        a, li = np.asarray(rows["action"], dtype=np.int64), np.asarray(rows["line"], dtype=np.int64)
        if a.size and (int(a.max()) >= n_action or int(li.max()) >= n_line):
            raise ValueError(f"{path} indexes beyond ({n_action} actions, {n_line} lines): built for another grid")
        relief = np.zeros((n_action, n_line), dtype=np.float32)
        frac = np.zeros((n_action, n_line), dtype=np.float32)
        relief[a, li] = np.asarray(rows["relief"], dtype=np.float32)
        frac[a, li] = np.asarray(rows["frac_pos"], dtype=np.float32)
        side = path.with_suffix(".json")
        meta = json.loads(side.read_text()) if side.exists() else {}
        meta["path"] = str(path)
        return cls(_frozen(relief), _frozen(frac), meta)

    def __repr__(self) -> str:
        n_pos = int((self.relief > 0).sum())
        return (
            f"RemedialCatalogue({self.relief.shape[0]} actions x {self.relief.shape[1]} lines,"
            f" {self.n_pairs} pairs, {n_pos} positive)"
        )
