"""The discrete backbone every ablation arm trains on: MaskablePPO over the
frozen ``set_bus`` candidate set, inside the frozen environment.

One backbone for seven arms. MaskablePPO is used even by ``Raw`` -- with the
legality mask only, which every arm gets (design doc 5.2) -- so that the only
thing that differs between arms is what the provider adds.

The saved directory is self-describing and a superset of what
``scripts/train_ppo.py`` writes, so ``grid2op_vkg.agents.load_ppo_agent`` still
understands the Raw layout and :func:`load_arm_agent` understands all of them:

    obs_attr_to_keep.json   base attributes + "vkg_feat" when the arm has state
    act_space.json          {"type": "discrete", "attr_to_keep", "n", "masked": true}
    run_meta.json           config sha, seed, budget, wrapper, timings
    vkg_arm.json            the arm's VkgConfig plus schema/action fingerprints
    .normalize_obs          the base attributes that were normalised (never vkg_feat)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from grid2op.gym_compat import BoxGymObsSpace, DiscreteActSpace

from grid2op_vkg.vkg.config import VkgConfig
from grid2op_vkg.vkg.obs_space import VKG_ATTR
from grid2op_vkg.vkg.runner import ArmSetup, build_arm


@dataclass
class DiscreteGym:
    """A gym env ready for MaskablePPO, plus what was written to describe it."""

    gym: Any
    setup: ArmSetup
    base_obs_attr: list[str]
    act_desc: dict[str, Any]


def build_discrete_gym(env: Any, project_cfg: dict, arm_cfg: VkgConfig, normalize_obs: bool = True) -> DiscreteGym:
    """Mirror of ``scripts/train_ppo.py::build_gym_env`` for a semantic arm.

    Differences, all deliberate:

    * the wrapper is the arm's ``SemanticGymEnv`` (reward shaping + masks),
      built with the frozen ``safe_max_rho``;
    * the observation space carries the arm's ``functs`` so the semantic block
      is part of the space rather than of the wrapper (see obs_space.py);
    * the semantic block is never normalised -- its entries are already in a
      known range, and ``normalize_attr`` would divide by the declared Box
      ceiling of 1000.
    """
    setup = build_arm(env, arm_cfg, project_cfg=project_cfg)
    base_obs_attr = [a for a in setup.obs_attr_to_keep if a != VKG_ATTR]

    g = setup.gymenv_class(env)
    g.observation_space.close()
    g.observation_space = BoxGymObsSpace(
        env.observation_space, attr_to_keep=setup.obs_attr_to_keep, **setup.obs_space_kwargs
    )
    if normalize_obs:
        for a in base_obs_attr:
            g.observation_space.normalize_attr(a)

    act_attr = list(setup.schema.require_actions().attr_to_keep)
    g.action_space.close()
    g.action_space = DiscreteActSpace(env.action_space, attr_to_keep=act_attr)
    n = int(g.action_space.n)
    if n != setup.schema.actions.n:
        raise RuntimeError(f"gym space has {n} actions, the schema's candidate set has {setup.schema.actions.n}")
    act_desc = {"type": "discrete", "attr_to_keep": act_attr, "n": n, "masked": True}
    return DiscreteGym(gym=g, setup=setup, base_obs_attr=base_obs_attr, act_desc=act_desc)


def write_model_dir(model_dir: Path, dg: DiscreteGym, meta: dict[str, Any]) -> None:
    model_dir.mkdir(parents=True, exist_ok=True)
    setup = dg.setup
    (model_dir / "obs_attr_to_keep.json").write_text(json.dumps(setup.obs_attr_to_keep))
    (model_dir / "act_space.json").write_text(json.dumps(dg.act_desc))
    (model_dir / ".normalize_obs").write_text("\n".join(dg.base_obs_attr))
    arm = {
        "config": setup.cfg.to_dict(),
        "schema_fingerprint": setup.schema.fingerprint,
        "action_set_fingerprint": setup.schema.require_actions().fingerprint,
        "feature_names": setup.provider.feature_names,
        "required_obs_attrs": sorted(setup.provider.required_obs_attrs),
    }
    (model_dir / "vkg_arm.json").write_text(json.dumps(arm, indent=1, ensure_ascii=False))
    (model_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))


def train_arm_discrete(
    project_cfg: dict,
    arm_cfg: VkgConfig,
    seed: int,
    name: str,
    save_path: str | Path = "checkpoints/arms",
    logs_dir: str | Path | None = "logs/arms",
    iterations: int | None = None,
    save_every: int | None = None,
    verbose: int = 1,
    norm_reward: bool = False,
    **ppo_kwargs: Any,
):
    """Train one arm; returns ``(model, model_dir, DiscreteGym)``."""
    import random

    import torch
    from sb3_contrib import MaskablePPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    from grid2op_vkg.config import make_env

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    tr = project_cfg["training"]
    iterations = int(iterations if iterations is not None else tr["total_agent_steps"])

    env = make_env(project_cfg, seed=seed)
    dg = build_discrete_gym(env, project_cfg, arm_cfg)
    model_dir = Path(save_path) / name
    meta = {
        "name": name,
        "arm": arm_cfg.arm,
        "backbone": "MaskablePPO",
        "config": project_cfg["_path"],
        "config_sha": project_cfg["_sha256"],
        "config_version": project_cfg.get("version"),
        "env": project_cfg["env"]["name"],
        "action_space": dg.act_desc,
        "obs_attr": dg.setup.obs_attr_to_keep,
        "wrapper": f"{dg.setup.gymenv_class.__module__}.{dg.setup.gymenv_class.__name__}",
        "safe_max_rho": project_cfg["heuristics"]["safe_max_rho"],
        "reward_cumul": dg.gym._reward_cumul,
        "iterations": iterations,
        "seed": seed,
        "net_arch": tr["net_arch"],
        "learning_rate": tr["learning_rate"],
        "started": datetime.now().isoformat(timespec="seconds"),
    }
    write_model_dir(model_dir, dg, meta)

    logs = None
    if logs_dir is not None:
        logs = Path(logs_dir) / name
        logs.mkdir(parents=True, exist_ok=True)

    # training.ppo_kwargs is shared by every learned variant (frozen config);
    # explicit call-site kwargs are probes and win over it
    shared = dict(tr.get("ppo_kwargs") or {})
    shared.update(ppo_kwargs)
    meta["ppo_kwargs"] = shared
    meta["norm_reward"] = bool(norm_reward)
    # With reward_cumul="sum" a gym step is worth ~60 (agent woken again) or
    # k*60 with k in 90..200 (a good action bought a quiet stretch), so returns
    # span two orders of magnitude and the critic never leaves
    # explained_variance == 0. Normalising the *training* return fixes the
    # scale; observations and the policy are untouched, so evaluation needs no
    # VecNormalize statistics and the arm loads exactly as before.
    train_env: Any = dg.gym
    if norm_reward:
        from stable_baselines3.common.monitor import Monitor
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        # SB3 only auto-wraps a *raw* env in Monitor; hand it a VecEnv and
        # ep_len_mean/ep_rew_mean vanish from the rollout log, so wrap here.
        monitored = Monitor(dg.gym)
        train_env = VecNormalize(
            DummyVecEnv([lambda: monitored]),
            norm_obs=False,
            norm_reward=True,
            gamma=float(shared.get("gamma", 0.99)),
        )
    model = MaskablePPO(
        "MlpPolicy",
        train_env,
        learning_rate=float(tr["learning_rate"]),
        policy_kwargs={"net_arch": list(tr["net_arch"])},
        tensorboard_log=str(logs) if logs else None,
        seed=seed,
        verbose=verbose,
        **shared,
    )
    callback = None
    if save_every:
        callback = CheckpointCallback(save_freq=save_every, save_path=str(model_dir), name_prefix=name)
    t0 = time.time()
    model.learn(total_timesteps=iterations, callback=callback)
    model.save(str(model_dir / name))

    meta["finished"] = datetime.now().isoformat(timespec="seconds")
    meta["train_wall_s"] = round(time.time() - t0, 1)
    meta["gymenv_summary"] = dg.gym.summary()
    meta["provider_timings"] = dg.setup.provider.timings()
    (model_dir / "run_meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=float))
    env.close()
    return model, model_dir, dg


def load_arm_config(model_dir: str | Path) -> VkgConfig:
    d = json.loads((Path(model_dir) / "vkg_arm.json").read_text())["config"]
    d["state_queries"] = tuple(d["state_queries"])
    return VkgConfig(**d)
