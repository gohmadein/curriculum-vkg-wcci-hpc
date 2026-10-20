"""Rebuild a trained PPO agent (scripts/train_ppo.py output) for evaluation.

The model directory is self-describing: ``act_space.json`` says whether the
policy acts in the discrete set_bus space or the continuous box space,
``obs_attr_to_keep.json`` lists the observation attributes and ``run_meta.json``
carries the heuristic wrapper and its threshold. Directories produced by the
upstream ``ppo_sb3.train`` (no ``act_space.json``) are still understood.

Upstream ``ppo_sb3.evaluate`` looks for the ``.normalize_*`` markers one level
too high and silently skips normalisation; this loader reads them from where
they are written.
"""

import json
from pathlib import Path

import numpy as np
from grid2op.gym_compat import BoxGymActSpace, BoxGymObsSpace, DiscreteActSpace
from gymnasium.spaces import Discrete

from grid2op_vkg.baselines.ppo_sb3.utils import SB3Agent


class _SB3Agent(SB3Agent):
    """SB3Agent whose network output is coerced to what the gym action space expects."""

    def get_act(self, gym_obs, reward, done):
        action, _ = self.nn_model.predict(gym_obs, deterministic=True)
        if isinstance(self.nn_model.action_space, Discrete):
            action = int(np.asarray(action).ravel()[0])  # DiscreteActSpace.from_gym wants a plain int
        return action


def _resolve_wrapper(name):
    import importlib

    if "." in name:
        mod, _, cls = name.rpartition(".")
        return getattr(importlib.import_module(mod), cls)
    from grid2op_vkg.baselines import utils as u

    return getattr(u, name)


def load_ppo_agent(env, model_dir, gymenv_class=None, gymenv_kwargs=None, iter_num=None):
    model_dir = Path(model_dir)
    name = model_dir.name
    meta_file = model_dir / "run_meta.json"
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    obs_attr = json.loads((model_dir / "obs_attr_to_keep.json").read_text())

    gym_obs = BoxGymObsSpace(env.observation_space, attr_to_keep=obs_attr)
    if (model_dir / ".normalize_obs").exists():
        for attr in obs_attr:
            gym_obs.normalize_attr(attr)

    act_file = model_dir / "act_space.json"
    if act_file.exists():
        act_desc = json.loads(act_file.read_text())
    else:  # upstream layout: continuous box over act_attr_to_keep.json
        act_desc = {"type": "box", "attr_to_keep": json.loads((model_dir / "act_attr_to_keep.json").read_text())}
    if act_desc["type"] == "discrete":
        gym_act = DiscreteActSpace(env.action_space, attr_to_keep=act_desc["attr_to_keep"])
    else:
        gym_act = BoxGymActSpace(env.action_space, attr_to_keep=act_desc["attr_to_keep"])
        if (model_dir / ".normalize_act").exists():
            for attr in act_desc["attr_to_keep"]:
                gym_act.normalize_attr(attr)

    if gymenv_class is None:
        gymenv_class = _resolve_wrapper(meta.get("wrapper", "GymEnvWithRecoWithDN"))
    if gymenv_kwargs is None:
        gymenv_kwargs = {
            "safe_max_rho": meta.get("safe_max_rho", 0.9),
            "reward_cumul": meta.get("reward_cumul", "init"),
        }
    gymenv = gymenv_class(env, **gymenv_kwargs)
    gymenv.action_space.close()
    gymenv.action_space = gym_act
    gymenv.observation_space.close()
    gymenv.observation_space = gym_obs

    return _SB3Agent(
        env.action_space, gym_act, gym_obs, nn_path=str(model_dir / name), gymenv=gymenv, iter_num=iter_num
    )
