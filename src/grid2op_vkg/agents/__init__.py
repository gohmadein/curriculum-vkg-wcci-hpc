"""Project-owned agents (as opposed to the vendored ones in ``grid2op_vkg.baselines``).

Importing this package registers the P0 baselines with ``grid2op_vkg.registry``.
"""

from grid2op_vkg.agents import builtin as _builtin  # noqa: F401  (registration side effect)
from grid2op_vkg.agents.ppo_loader import load_ppo_agent
from grid2op_vkg.agents.reconnect import ReconnectAgent

__all__ = ["ReconnectAgent", "load_ppo_agent"]
