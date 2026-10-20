"""Compiled numpy backend -- the path the RL training loop actually runs.

Every kernel is built once, at construction, against the fixed
:class:`~grid2op_vkg.vkg.schema.GridSchema`, so per-step work is pure array
arithmetic. Measured on case14 this is roughly three orders of magnitude
cheaper than the SPARQL reference, which is what makes 1e6-step training runs
affordable without changing what the queries mean.
"""

from __future__ import annotations

import time
from collections.abc import Iterable

import numpy as np

from grid2op_vkg.vkg.backends.base import VkgBackend
from grid2op_vkg.vkg.queries.base import Kernel, Query
from grid2op_vkg.vkg.schema import GridSchema
from grid2op_vkg.vkg.state import GridState


class CompiledBackend(VkgBackend):
    """Answers queries by calling their compiled kernels."""

    name = "compiled"

    def __init__(self, schema: GridSchema, queries: Iterable[Query]) -> None:
        super().__init__(schema, queries)
        self._kernels: dict[str, Kernel] = {name: q.kernel(schema) for name, q in self.queries.items()}

    def _prepare(self, state: GridState) -> None:
        # Nothing to materialise. The derived views on GridState are lazy, so
        # the reasoning cost (adjacency) is paid by whichever query asks first
        # and is reported under that query rather than separately.
        self.timings.prepare_ms.append(0.0)
        self.timings.reason_ms.append(0.0)

    def _run(self, query: Query, state: GridState) -> np.ndarray:
        t0 = time.perf_counter()
        res = self._kernels[query.name](state)
        self.timings.record_query(query.name, (time.perf_counter() - t0) * 1e3)
        return np.asarray(res)
