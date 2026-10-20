"""The facade the rest of the codebase talks to.

One object owns the schema, the query set, the backend and the per-step
bookkeeping, so that the gym wrapper, the observation space and the agent all
see the same answers for the same step without recomputing anything.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from grid2op_vkg.vkg.backends import BACKENDS, VkgBackend
from grid2op_vkg.vkg.config import VkgConfig
from grid2op_vkg.vkg.queries.base import Query
from grid2op_vkg.vkg.queries.registry import build_queries
from grid2op_vkg.vkg.schema import GridSchema
from grid2op_vkg.vkg.schema import GridSchema as _GS  # noqa: F401  (re-exported for type checkers)
from grid2op_vkg.vkg.state import GridState

#: Stand-in for an unbounded feature when declaring a gym Box. It bounds the
#: *declared space*, never the values: clipping a semantic feature would be a
#: silent behaviour change, while a loose box is merely loose.
BOX_BOUND = 1.0e3


def default_query_spec(cfg: VkgConfig) -> dict[str, dict[str, Any]]:
    """The default query set, parameterised from the arm's shared thresholds.

    Built from ``cfg`` rather than from constants so that the Handcrafted arm,
    which reads the same ``cfg`` fields, is guaranteed to be answering the same
    question. Action queries are included only when the schema will carry a
    candidate action set -- that is checked at engine construction.
    """
    tau, k, adj = cfg.risk_tau, cfg.relevance_k, cfg.relevance_adjacency
    return {
        "sub_is_split": {},
        "controllable_subs": {},
        "reconnectable_lines": {},
        "line_adjacency": {},
        "lines_sharing_substation": {},
        "lines_at_risk": {"tau": tau},
        "lines_near_risk": {"tau": tau, "k": k, "adjacency": adj},
        "line_margin": {"tau": cfg.margin_tau},
        "overload_pressure": {"tau": tau},
        "headroom_near_risk": {"tau": tau, "k": k, "adjacency": adj},
        "legal_actions": {},
        "relevant_actions": {
            "tau": tau,
            "k": k,
            "adjacency": adj,
            "remedial_theta": cfg.remedial_theta if cfg.remedial_membership else -1.0,
        },
        # knowledge layer: the remedial-action catalogue (empty when none is attached)
        "remedial_actions": {"tau": tau, "theta": cfg.remedial_theta},
        "remedial_score": {"tau": tau},
        "predicted_max_rho": {},
        "islanding_actions": {},
        # budget layer: ordering and affordability for a search that cannot
        # simulate every candidate (results/eval_arms/README.md)
        "action_disruption": {},
        "priority_substations": {},
        "affordable_actions": {"max_split_subs": cfg.max_split_subs},
        "recovery_actions": {},
    }


class VkgEngine:
    """Schema + queries + backend, driven one environment step at a time.

    Usage::

        eng = VkgEngine.from_env(env, cfg)
        eng.begin_step(obs)
        eng.ask("lines_at_risk")     # base name is enough
        eng.feature_vector()
        eng.potential()
    """

    def __init__(self, schema: GridSchema, cfg: VkgConfig, queries: list[Query] | None = None) -> None:
        self.schema = schema
        self.cfg = cfg
        if queries is None:
            spec = cfg.queries if cfg.queries is not None else default_query_spec(cfg)
            if schema.actions is None:
                spec = {k: v for k, v in spec.items() if not k.endswith("_actions")}
            qs = build_queries(spec)
        else:
            qs = list(queries)
        self.queries = {q.base_name: q for q in qs}
        if len(self.queries) != len(qs):
            raise ValueError("two queries share a base_name; they would shadow each other")

        backend_cls = BACKENDS.get(cfg.backend)
        if backend_cls is None:
            raise KeyError(f"unknown backend {cfg.backend!r}; have {sorted(BACKENDS)}")
        self.backend: VkgBackend = backend_cls(schema, qs)

        self._step_ms: list[float] = []
        self._t0: float | None = None
        self._n_steps = 0
        self._token: tuple[int, int] | None = None
        self._last_obs: Any = None
        self.state: GridState | None = None
        self._feature_layout = self._build_feature_layout()

    @classmethod
    def from_env(cls, env: Any, cfg: VkgConfig, env_name: str | None = None) -> VkgEngine:
        return cls(GridSchema.from_env(env, env_name), cfg)

    # ------------------------------------------------------------------ #
    # per step
    # ------------------------------------------------------------------ #
    def begin_step(self, obs: Any) -> GridState:
        """Install the observation. Everything else is lazy from here.

        Idempotent for a given observation. The state and reward channels are
        driven from different places -- the observation space and the gym
        wrapper -- and both announce the step; without this the VKG-Full arm
        would evaluate every query twice and report double the query count.
        """
        # Identity, not id(): an id() is only unique among *live* objects. Building
        # a BoxGymObsSpace probes the feature function with a template observation
        # (step 0, NaN loadings); once that template is freed, the first real
        # observation after reset() can be allocated at the same address with the
        # same step, and an id()-keyed token then served the template's NaN state.
        # Holding a reference to the observation keeps its address from being reused.
        step = int(getattr(obs, "current_step", -1))
        token = (id(obs), step)
        if obs is self._last_obs and token == self._token:
            return self.state
        self._t0 = time.perf_counter()
        state = GridState.from_obs(self.schema, obs)
        self.state = state
        self.backend.begin_step(state)
        self._n_steps += 1
        self._token = token
        self._last_obs = obs
        return state

    def end_step(self) -> float:
        """Close the timing window. Returns the step's VKG cost in ms."""
        if self._t0 is None:
            return 0.0
        ms = (time.perf_counter() - self._t0) * 1e3
        self._step_ms.append(ms)
        self._t0 = None
        return ms

    def ask(self, name: str) -> np.ndarray:
        """Answer a query by base name (``lines_at_risk``) or full name."""
        if name in self.queries:
            return self.backend.ask(self.queries[name].name)
        return self.backend.ask(name)

    # ------------------------------------------------------------------ #
    # VKG-State: the observation block
    # ------------------------------------------------------------------ #
    def _element_labels(self, dim: int) -> list[str]:
        s = self.schema
        if dim == s.n_line:
            return list(s.name_line)
        if dim == s.n_sub:
            return list(s.name_sub)
        if dim == s.n_gen:
            return list(s.name_gen)
        return [str(i) for i in range(dim)]

    def _build_feature_layout(self) -> list[tuple[str, int, float, float]]:
        """``(base_name, size, low, high)`` for each query feeding the state.

        Empty unless the arm has a state channel: a reward-only arm has no
        reason to carry the state queries in its active set, and demanding them
        would force every arm to share one query set.
        """
        if not self.cfg.state:
            return []
        layout = []
        for base in self.cfg.state_queries:
            q = self.queries.get(base)
            if q is None:
                raise KeyError(f"state_queries names {base!r} but it is not in the active query set")
            sig = q.sig(self.schema)
            if len(sig.shape) > 1:
                raise ValueError(
                    f"query {base!r} is {sig.shape}-shaped; matrix-valued queries do not belong "
                    "in a flat observation block (feed them to the GNN backbone instead)"
                )
            layout.append((base, sig.size, sig.low, sig.high))
        return layout

    @property
    def feature_dim(self) -> int:
        return sum(size for _, size, _, _ in self._feature_layout)

    @property
    def feature_names(self) -> list[str]:
        names = []
        for base, size, _, _ in self._feature_layout:
            if size == 1:
                names.append(base)
            else:
                names.extend(f"{base}[{lab}]" for lab in self._element_labels(size))
        return names

    def feature_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Finite (low, high) for declaring a gym Box. See :data:`BOX_BOUND`."""
        low, high = [], []
        for _, size, lo, hi in self._feature_layout:
            low.extend([float(np.clip(lo, -BOX_BOUND, BOX_BOUND))] * size)
            high.extend([float(np.clip(hi, -BOX_BOUND, BOX_BOUND))] * size)
        return np.asarray(low, dtype=np.float32), np.asarray(high, dtype=np.float32)

    def feature_vector(self) -> np.ndarray:
        """Concatenated query results, in the fixed order of the layout."""
        parts = []
        for base, size, _, _ in self._feature_layout:
            val = np.asarray(self.ask(base), dtype=np.float32).reshape(-1)
            if val.size != size:
                raise ValueError(f"query {base!r} returned {val.size} values, layout says {size}")
            parts.append(val)
        return np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)

    # ------------------------------------------------------------------ #
    # VKG-Reward: the shaping potential
    # ------------------------------------------------------------------ #
    def potential(self) -> float:
        """Phi(s), reading only the current observation.

        Negated pressure: a grid with less overload has a higher potential, so
        moving towards safety earns positive shaping. Phi must never consult a
        future state -- that is what keeps the shaping potential-based and
        therefore policy-invariant.
        """
        return -float(np.asarray(self.ask(self.cfg.potential_query)).reshape(()))

    # ------------------------------------------------------------------ #
    # VKG-Mask: the semantic filter over the candidate set
    # ------------------------------------------------------------------ #
    def semantic_mask(self) -> np.ndarray:
        """relevant AND NOT islanding. Legality is shared plumbing, not here."""
        return np.asarray(self.ask("relevant_actions")) & ~np.asarray(self.ask("islanding_actions"))

    # ------------------------------------------------------------------ #
    # reporting
    # ------------------------------------------------------------------ #
    def timings(self) -> dict[str, float]:
        out = dict(self.backend.timings.summary())
        if self._step_ms:
            out["vkg_step_ms_p50"] = float(np.percentile(self._step_ms, 50))
            out["vkg_step_ms_p95"] = float(np.percentile(self._step_ms, 95))
            out["vkg_step_ms_max"] = float(np.max(self._step_ms))
        out["n_steps"] = self._n_steps
        out["backend"] = self.backend.name
        return out

    def reset_timings(self) -> None:
        self.backend.timings.reset()
        self._step_ms.clear()
        self._n_steps = 0

    def invalidate(self) -> None:
        """Force the next :meth:`begin_step` to recompute, whatever it is given."""
        self._token = None
        self._last_obs = None

    @property
    def required_obs_attrs(self) -> frozenset[str]:
        """Union of what every active query reads. The fairness guard's input."""
        out: frozenset[str] = frozenset()
        for q in self.queries.values():
            out |= q.required_obs_attrs
        return out

    def close(self) -> None:
        self.backend.close()

    def __repr__(self) -> str:
        return (
            f"<VkgEngine arm={self.cfg.arm} backend={self.backend.name} "
            f"{len(self.queries)} queries, {self.feature_dim} features>"
        )
