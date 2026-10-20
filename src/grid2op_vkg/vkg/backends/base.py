"""The backend contract: answer a query against a state, and say what it cost."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from grid2op_vkg.vkg.queries.base import Query
from grid2op_vkg.vkg.schema import GridSchema
from grid2op_vkg.vkg.state import GridState


@dataclass
class Timings:
    """Per-phase wall clock, in milliseconds, accumulated over a run.

    Kept separate for reasoning and querying because the paper has to report
    them separately (baseline survey section 9).
    """

    prepare_ms: list[float] = field(default_factory=list)
    reason_ms: list[float] = field(default_factory=list)
    query_ms: dict[str, list[float]] = field(default_factory=dict)

    def record_query(self, name: str, ms: float) -> None:
        self.query_ms.setdefault(name, []).append(ms)

    def summary(self) -> dict[str, float]:
        def pct(xs, p):
            return float(np.percentile(xs, p)) if xs else 0.0

        per_step = [a + b for a, b in zip(self.prepare_ms, self.reason_ms, strict=False)]
        out = {
            "prepare_ms_p50": pct(self.prepare_ms, 50),
            "prepare_ms_p95": pct(self.prepare_ms, 95),
            "reason_ms_p50": pct(self.reason_ms, 50),
            "reason_ms_p95": pct(self.reason_ms, 95),
            "setup_ms_p50": pct(per_step, 50),
            "n_queries": sum(len(v) for v in self.query_ms.values()),
        }
        all_q = [ms for v in self.query_ms.values() for ms in v]
        out["query_ms_p50"] = pct(all_q, 50)
        out["query_ms_p95"] = pct(all_q, 95)
        return out

    def reset(self) -> None:
        self.prepare_ms.clear()
        self.reason_ms.clear()
        self.query_ms.clear()


class VkgBackend(ABC):
    """Executes queries. Implementations differ only in speed, never in answers.

    The whole design rests on that promise, so it is a test
    (``tests/test_vkg_queries.py``), not a comment.
    """

    name: str = "abstract"

    def __init__(self, schema: GridSchema, queries: Iterable[Query]) -> None:
        self.schema = schema
        self.queries: dict[str, Query] = {q.name: q for q in queries}
        self.timings = Timings()
        self._state: GridState | None = None
        self._results: dict[str, np.ndarray] = {}
        self._static_results: dict[str, np.ndarray] = {}

    def begin_step(self, state: GridState) -> None:
        """Install a new state and drop the previous step's answers."""
        self._state = state
        self._results.clear()
        self._prepare(state)

    def ask(self, name: str) -> np.ndarray:
        """Answer ``name`` for the current state, at most once per step."""
        if self._state is None:
            raise RuntimeError("call begin_step() before ask()")
        if name in self._results:
            return self._results[name]
        if name not in self.queries:
            raise KeyError(f"query {name!r} not registered; have {sorted(self.queries)}")
        query = self.queries[name]
        if query.static and name in self._static_results:
            res = self._static_results[name]
        else:
            res = self._run(query, self._state)
            res.flags.writeable = False
            if query.static:
                self._static_results[name] = res
        self._results[name] = res
        return res

    def ask_all(self) -> dict[str, np.ndarray]:
        return {name: self.ask(name) for name in self.queries}

    # -- implementation hooks ----------------------------------------- #
    def _prepare(self, state: GridState) -> None:  # noqa: B027  optional hook, not abstract
        """Per-step setup: materialisation, inference. No-op for compiled kernels."""

    @abstractmethod
    def _run(self, query: Query, state: GridState) -> np.ndarray: ...

    def close(self) -> None:  # noqa: B027  optional hook, not abstract
        """Release external resources. Only the Ontop backend will have any."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {len(self.queries)} queries>"
