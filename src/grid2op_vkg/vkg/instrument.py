"""Budgets and meters.

The fairness protocol in the design documents is a list of things that must be
equal across arms. A list in a document drifts; a counter that raises does not.
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from dataclasses import dataclass, field

import numpy as np


class SimulateBudgetExceeded(RuntimeError):
    """Raised when an arm spends more simulator calls than the protocol allows."""


@dataclass
class SimulateBudget:
    """Counts, and optionally caps, ``obs.simulate()`` calls.

    ``obs.simulate()`` is a full power-flow solve: it is the knob that lets a
    rule-based agent buy performance with compute. The baseline survey is
    explicit that it has to be inside the budget when methods are compared, so
    the VKG side defaults to **zero** -- queries read the observation, they do
    not run the simulator.

    Patches ``BaseObservation.simulate`` globally, matching the approach in
    ``scripts/eval_baselines.py`` so both sides report the same quantity.
    """

    per_step: int = 0
    enforce: bool = True

    n_calls: int = 0
    n_steps: int = 0
    _this_step: int = 0
    _orig = None

    def __enter__(self) -> SimulateBudget:
        from grid2op.Observation import BaseObservation

        budget = self
        self._orig = BaseObservation.simulate

        def counted(obs_self, *args, **kwargs):
            budget.n_calls += 1
            budget._this_step += 1
            if budget.enforce and budget._this_step > budget.per_step:
                raise SimulateBudgetExceeded(
                    f"obs.simulate() called {budget._this_step} times in one step, "
                    f"budget is {budget.per_step}. Either the arm is cheating or the "
                    "budget in the fairness contract needs raising for every arm."
                )
            return budget._orig(obs_self, *args, **kwargs)

        BaseObservation.simulate = counted
        return self

    def __exit__(self, *exc) -> None:
        from grid2op.Observation import BaseObservation

        if self._orig is not None:
            BaseObservation.simulate = self._orig
            self._orig = None

    def new_step(self) -> None:
        self._this_step = 0
        self.n_steps += 1

    @property
    def calls_per_step(self) -> float:
        return self.n_calls / self.n_steps if self.n_steps else 0.0


@dataclass
class StepTimer:
    """Per-decision wall clock, with a ceiling the protocol can check.

    Exceeding the limit is recorded rather than raised: a single slow step is a
    data point, and killing the run would lose the rest of the episode. The
    violation rate is what gets reported.
    """

    limit_ms: float = 1000.0
    samples: list[float] = field(default_factory=list)
    n_violations: int = 0

    @contextmanager
    def measure(self):
        t0 = time.perf_counter()
        try:
            yield
        finally:
            ms = (time.perf_counter() - t0) * 1e3
            self.samples.append(ms)
            if ms > self.limit_ms:
                self.n_violations += 1

    def summary(self, prefix: str = "step") -> dict[str, float]:
        if not self.samples:
            return {}
        xs = np.asarray(self.samples)
        return {
            f"{prefix}_ms_p50": float(np.percentile(xs, 50)),
            f"{prefix}_ms_p95": float(np.percentile(xs, 95)),
            f"{prefix}_ms_max": float(xs.max()),
            f"{prefix}_timeout_rate": self.n_violations / xs.size,
        }

    def reset(self) -> None:
        self.samples.clear()
        self.n_violations = 0
