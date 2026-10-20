"""Registry entries for the P0 baselines. Imported by ``grid2op_vkg.agents``."""

from grid2op_vkg.registry import register_agent


@register_agent("DoNothing")
def _do_nothing(env, cfg, seed, **_):
    from grid2op_vkg.baselines.do_nothing import DoNothing

    return DoNothing(env.action_space, env.observation_space, "DoNothing")


@register_agent("Reconnect")
def _reconnect(env, cfg, seed, **_):
    from grid2op_vkg.agents.reconnect import ReconnectAgent

    return ReconnectAgent(env.action_space)


@register_agent("ExpertAgent")
def _expert(env, cfg, seed, grid_name="IEEE14", **_):
    from grid2op_vkg.baselines.expert_agent import ExpertAgent

    return ExpertAgent(env.action_space, env.observation_space, "ExpertAgent", gridName=grid_name)


@register_agent("OptimCVXPY")
def _optim(env, cfg, seed, rho_danger=0.95, rho_safe=0.85, **kw):
    from grid2op_vkg.baselines.optim_cvxpy import OptimCVXPY

    return OptimCVXPY(env.action_space, env, rho_danger=rho_danger, rho_safe=rho_safe, **kw)


@register_agent("PPO")
def _ppo(env, cfg, seed, model_dir=None, **_):
    import torch

    from grid2op_vkg.agents.ppo_loader import load_ppo_agent

    if model_dir is None:
        raise ValueError("PPO needs model_dir=<save_path>/<name>")
    torch.set_num_threads(1)
    return load_ppo_agent(env, model_dir)
