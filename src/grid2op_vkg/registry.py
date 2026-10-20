"""Agent registry: the hook other packages use to plug agents into the harness.

Register a factory once, anywhere that gets imported::

    from grid2op_vkg.registry import register_agent

    @register_agent("VKG-Reward")
    def _build(env, cfg, seed, **kwargs):
        return MyAgent(env.action_space, ...)

Then ``scripts/eval_baselines.py --agent-module my_pkg.agents --agents VKG-Reward``
will build it inside every worker with the frozen config. Factories receive the
grid2op env (already built from ``configs/base.yaml``), the loaded config dict,
the run seed and any per-agent kwargs passed on the command line.
"""

from collections.abc import Callable

AgentFactory = Callable[..., object]

_FACTORIES: dict[str, AgentFactory] = {}


def register_agent(name: str) -> Callable[[AgentFactory], AgentFactory]:
    def deco(fn: AgentFactory) -> AgentFactory:
        if name in _FACTORIES and _FACTORIES[name] is not fn:
            raise ValueError(f"agent {name!r} is already registered by {_FACTORIES[name].__module__}")
        _FACTORIES[name] = fn
        return fn

    return deco


def make_agent(name: str, env, cfg: dict, seed: int, **kwargs):
    try:
        factory = _FACTORIES[name]
    except KeyError:
        raise KeyError(f"unknown agent {name!r}; registered: {sorted(_FACTORIES)}") from None
    agent = factory(env, cfg, seed, **kwargs)
    if hasattr(agent, "seed"):
        agent.seed(seed)
    return agent


def available() -> list[str]:
    return sorted(_FACTORIES)
