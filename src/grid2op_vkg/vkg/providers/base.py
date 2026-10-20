"""The interface Handcrafted and VKG share.

This is the most load-bearing abstraction in the ablation. Everything
downstream -- the gym wrapper, the observation space, the masking -- is written
against :class:`FeedbackProvider` and never against a concrete implementation,
so the ``Handcrafted`` and ``VKG-Full`` arms run byte-identical plumbing and
differ only in where the numbers come from. Any difference in their scores is
then attributable to the mechanism rather than to the wiring.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class SearchGuidance:
    """What a budget-limited search needs to know that only the grid can say.

    Every field is a data-layer fact, so it comes from the provider. The search
    that consumes them -- group, interleave, stop at the budget -- mentions no
    grid vocabulary at all, which is what keeps it untouched by a schema change.

    disruption:
        per candidate action, how many live terminals it actually reassigns.
    priority_subs:
        per substation, whether it sits at an end of a most-loaded line.
    affordable:
        per action, whether it avoids taking on topological debt we cannot
        service (see ``AffordableActions``).
    recovery:
        per action, whether it restores a split substation to busbar 1.
    split:
        per substation, whether it is currently split.
    reconnectable:
        per line, whether it is out of service with its cooldown expired.
        Reconnection is free -- it costs no simulation and is never wrong --
        but the candidate action set is ``set_bus`` only under the frozen
        config, so the search cannot find it by masking and has to be told.
    at_risk:
        per line, whether it is at or above the risk threshold. The search uses
        only ``.any()`` -- which regime it is in -- but the per-line answer is
        what the query returns and what the two providers are compared on.
    """

    disruption: np.ndarray
    priority_subs: np.ndarray
    affordable: np.ndarray
    recovery: np.ndarray
    split: np.ndarray
    reconnectable: np.ndarray
    at_risk: np.ndarray
    #: knowledge layer (all-False / 0 unless a remedial catalogue is attached to
    #: the schema): per action, whether the catalogue says it relieves a line at
    #: risk by at least ``remedial_theta``, and the largest such relief
    remedial: np.ndarray
    remedial_score: np.ndarray


class FeedbackProvider(ABC):
    """Produces the four semantic feedback channels for one environment step.

    A provider that does not implement a channel returns ``None`` for it; the
    consumer treats that as "this arm has the channel switched off".
    """

    #: Identifies the provider in logs and results.
    name: str = "abstract"

    #: Observation attributes this provider reads, directly or through its
    #: queries. Checked against the Raw arm's observation list by
    #: :mod:`grid2op_vkg.vkg.fairness` -- a provider may never see more than the
    #: baseline does.
    required_obs_attrs: frozenset[str] = frozenset()

    #: obs.simulate() calls this provider makes per step. Must stay at 0 unless
    #: every arm is granted the same budget: the simulator is compute, and
    #: compute is exactly what the fairness protocol equalises.
    simulate_calls_per_step: int = 0

    # ------------------------------------------------------------------ #
    @abstractmethod
    def begin_step(self, obs: Any) -> None:
        """Install a new observation. Called once per agent decision."""

    def end_step(self) -> float:
        """Close the timing window; return this step's cost in milliseconds."""
        return 0.0

    # -- the four channels -------------------------------------------- #
    def features(self) -> np.ndarray | None:
        """VKG-State: extra observation entries, or ``None``."""
        return None

    def potential(self) -> float | None:
        """VKG-Reward: Phi(s) for potential-based shaping, or ``None``.

        Must be a function of the current and past observation only. Reading a
        future state would break policy invariance, which is the entire reason
        the shaping is potential-based.
        """
        return None

    def action_mask(self, action_set: Any) -> np.ndarray | None:
        """VKG-Mask: boolean over the candidate action set, or ``None``."""
        return None

    def search_guidance(self) -> SearchGuidance | None:
        """Ordering and affordability facts for a budget-limited search.

        Not one of the four ablation channels: it feeds the rule-based agents
        rather than the learner, and it is the role the measurements actually
        support (see results/eval_arms/README.md). ``None`` means this arm
        offers no guidance and a consumer must fall back to its own ordering.
        """
        return None

    def justify(self, action_idx: int) -> np.ndarray | None:
        """Provenance for one admitted action: (n_line, n_sub) witness matrix.

        ``[r, s]`` is True when line ``r`` is at risk, substation ``s`` is an
        endpoint of a line within the relevance radius of ``r``, and the action
        reconfigures ``s`` -- the facts that put the action on the table. On
        the VKG side this is a query over the same graph the decision came from
        (``ExplainAction``); ``None`` means the provider offers no provenance.
        """
        return None

    def aux_targets(self) -> dict[str, np.ndarray] | None:
        """VKG-Aux: auxiliary prediction targets, or ``None``."""
        return None

    # -- reporting ------------------------------------------------------ #
    @property
    def feature_names(self) -> list[str]:
        return []

    def feature_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return np.zeros(0, dtype=np.float32), np.zeros(0, dtype=np.float32)

    @property
    def feature_dim(self) -> int:
        return len(self.feature_names)

    def timings(self) -> dict[str, float]:
        return {}

    def close(self) -> None:  # noqa: B027  optional hook, not abstract
        """Release resources. Only backend-backed providers have any."""

    def __repr__(self) -> str:
        return f"<{type(self).__name__} {self.feature_dim} features>"
