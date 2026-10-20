"""Registry entries for the seven ablation arms.

    uv run scripts/eval_baselines.py --agent-module grid2op_vkg.vkg.agents \\
        --agents Raw VKG-Full --agent-kwargs '{"Raw": {"model_dir": "..."}, ...}'

Each factory rebuilds the arm exactly as it was trained -- same frozen config,
same provider, same observation block, same candidate set (fingerprints are
checked) -- and returns a grid2op agent whose policy is queried *with the action
mask*, since a MaskablePPO policy evaluated without its mask is a different
policy.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace

from grid2op_vkg.baselines.ppo_sb3.utils import SB3Agent
from grid2op_vkg.registry import register_agent
from grid2op_vkg.vkg.config import ARM_NAMES
from grid2op_vkg.vkg.obs_space import VKG_ATTR
from grid2op_vkg.vkg.ppo import load_arm_config
from grid2op_vkg.vkg.runner import build_arm


class MaskedSB3Agent(SB3Agent):
    """SB3Agent that hands the wrapper's mask to ``predict``.

    ``GymAgent.act`` runs the heuristics and then calls :meth:`get_act` with the
    converted observation only; the raw grid2op observation is stashed in
    :meth:`act` so the mask can be computed from the very state being decided.
    """

    def load(self):
        """Same as upstream ``SB3Agent.load`` but through MaskablePPO, whose policy ``PPO.load`` rejects."""
        from sb3_contrib import MaskablePPO

        custom_objects = {"action_space": self._gym_act_space, "observation_space": self._gym_obs_space}
        custom_objects.update(self.custom_load_dict)
        path = self._nn_path if self._iter_num is None else f"{self._nn_path}_{self._iter_num}_steps"
        self.nn_model = MaskablePPO.load(path, custom_objects=custom_objects)

    def act(self, observation, reward, done):
        self._current_obs = observation
        return super().act(observation, reward, done)

    def get_act(self, gym_obs, reward, done):
        mask = self.gymenv.mask_for(self._current_obs)
        action, _ = self.nn_model.predict(gym_obs, deterministic=True, action_masks=mask)
        return int(np.asarray(action).ravel()[0])


def load_arm_agent(env, model_dir, project_cfg: dict, iter_num=None) -> MaskedSB3Agent:
    model_dir = Path(model_dir)
    name = model_dir.name
    arm_cfg = load_arm_config(model_dir)
    saved = json.loads((model_dir / "vkg_arm.json").read_text())
    obs_attr = json.loads((model_dir / "obs_attr_to_keep.json").read_text())
    base_attr = [a for a in obs_attr if a != VKG_ATTR]

    setup = build_arm(env, arm_cfg, project_cfg=project_cfg, obs_attr_to_keep=base_attr)
    if setup.schema.fingerprint != saved["schema_fingerprint"]:
        raise RuntimeError(
            f"{model_dir}: trained on schema {saved['schema_fingerprint']}, env is {setup.schema.fingerprint}"
        )
    if setup.obs_attr_to_keep != obs_attr:
        raise RuntimeError(f"{model_dir}: observation layout changed: {obs_attr} vs {setup.obs_attr_to_keep}")

    gym_obs = BoxGymObsSpace(env.observation_space, attr_to_keep=obs_attr, **setup.obs_space_kwargs)
    normalised = (model_dir / ".normalize_obs").read_text().split() if (model_dir / ".normalize_obs").exists() else []
    for a in normalised:
        gym_obs.normalize_attr(a)
    act_desc = json.loads((model_dir / "act_space.json").read_text())
    gym_act = DiscreteActSpace(env.action_space, attr_to_keep=act_desc["attr_to_keep"])
    if int(gym_act.n) != act_desc["n"]:
        raise RuntimeError(f"{model_dir}: candidate set has {gym_act.n} actions, trained with {act_desc['n']}")

    gymenv = setup.gymenv_class(env)
    gymenv.action_space.close()
    gymenv.action_space = gym_act
    gymenv.observation_space.close()
    gymenv.observation_space = gym_obs

    agent = MaskedSB3Agent(
        env.action_space, gym_act, gym_obs, nn_path=str(model_dir / name), gymenv=gymenv, iter_num=iter_num
    )
    agent.setup = setup
    return agent


def _make_factory(arm: str):
    def factory(env, cfg, seed, model_dir=None, iter_num=None, **_):
        import torch

        if model_dir is None:
            raise ValueError(f"{arm} needs model_dir=<save_path>/<name>")
        torch.set_num_threads(1)
        # iter_num picks an intermediate CheckpointCallback file, i.e. a point
        # on the learning curve rather than the final policy
        agent = load_arm_agent(env, model_dir, cfg, iter_num=iter_num)
        if agent.setup.cfg.arm != arm:
            raise ValueError(f"{model_dir} was trained as arm {agent.setup.cfg.arm!r}, requested {arm!r}")
        return agent

    factory.__name__ = f"_build_{arm.replace('-', '_')}"
    return factory


for _arm in ARM_NAMES:
    register_agent(_arm)(_make_factory(_arm))

# the rule-based agents register themselves on import
from grid2op_vkg.vkg import budget_baselines as _budget  # noqa: E402, F401
from grid2op_vkg.vkg import greedy as _greedy  # noqa: E402, F401
from grid2op_vkg.vkg import guard as _guard  # noqa: E402, F401
from grid2op_vkg.vkg import hybrid as _hybrid  # noqa: E402, F401
