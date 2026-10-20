"""The Raw arm's provider: present in the plumbing, absent from the signal.

Raw could have been "no provider at all", but then the Raw arm would take a
different code path from every other arm and would no longer be a clean
control. Instead it takes the same path and gets ``None`` from every channel.
"""

from __future__ import annotations

from typing import Any

from grid2op_vkg.vkg.providers.base import FeedbackProvider


class NullProvider(FeedbackProvider):
    """Reads nothing, returns nothing, costs nothing."""

    name = "none"
    required_obs_attrs = frozenset()
    simulate_calls_per_step = 0

    def begin_step(self, obs: Any) -> None:
        return None
