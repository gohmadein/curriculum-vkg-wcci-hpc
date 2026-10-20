"""Wiring an ablation arm into the vendored PPO baseline.

Nothing in ``src/grid2op_vkg/baselines/`` is modified. Everything happens
through the three extension points the upstream code already exposes:
``gymenv_class``, ``gymenv_kwargs`` and ``obs_space_kwargs``.
"""

from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass
from typing import Any

from grid2op_vkg.vkg import fairness, obs_space
from grid2op_vkg.vkg.actions import DEFAULT_ACT_ATTRS
from grid2op_vkg.vkg.config import FairnessContract, VkgConfig
from grid2op_vkg.vkg.gymenv import make_gymenv_class
from grid2op_vkg.vkg.providers import make_provider
from grid2op_vkg.vkg.providers.base import FeedbackProvider
from grid2op_vkg.vkg.schema import GridSchema

#: Fallback observation list when no project config is passed. Identical to
#: ``observation.attr_to_keep`` in configs/base.yaml (a test checks that); the
#: VKG arms derive everything they need from these fields, so parity is kept
#: without widening the baseline.
BASE_OBS_ATTRS: tuple[str, ...] = (
    "day_of_week",
    "hour_of_day",
    "minute_of_hour",
    "prod_p",
    "prod_v",
    "load_p",
    "load_q",
    "actual_dispatch",
    "target_dispatch",
    "topo_vect",
    "time_before_cooldown_line",
    "time_before_cooldown_sub",
    "rho",
    "timestep_overflow",
    "line_status",
)


@dataclass
class ArmSetup:
    """Everything ``train()`` and ``evaluate()`` need for one arm."""

    cfg: VkgConfig
    schema: GridSchema
    provider: FeedbackProvider
    gymenv_class: type
    obs_attr_to_keep: list[str]
    obs_space_kwargs: dict[str, Any]
    contract: FairnessContract

    def train_kwargs(self) -> dict[str, Any]:
        """Keyword arguments to splat into ``ppo_sb3.train``."""
        return {
            "gymenv_class": self.gymenv_class,
            "gymenv_kwargs": {},
            "obs_attr_to_keep": self.obs_attr_to_keep,
            "obs_space_kwargs": self.obs_space_kwargs,
        }

    def evaluate_kwargs(self) -> dict[str, Any]:
        """Keyword arguments to splat into ``ppo_sb3.evaluate``.

        ``obs_attr_to_keep`` is deliberately absent: ``evaluate()`` restores it
        from the JSON that ``train()`` wrote, which is what keeps the two in
        step. Only the functs have to be handed over again, because a Python
        closure cannot be serialised into that JSON.
        """
        return {
            "gymenv_class": self.gymenv_class,
            "gymenv_kwargs": {},
            "obs_space_kwargs": self.obs_space_kwargs,
        }


def build_arm(
    env: Any,
    cfg: VkgConfig,
    contract: FairnessContract | None = None,
    obs_attr_to_keep: list[str] | None = None,
    env_name: str | None = None,
    check_fairness: bool = True,
    project_cfg: dict | None = None,
    **provider_kwargs: Any,
) -> ArmSetup:
    """Assemble one ablation arm.

    ``project_cfg`` is the loaded ``configs/base.yaml``; when given, the
    observation list, the candidate action attributes and the heuristic
    threshold all come from it, so the arm is built inside the frozen setup
    rather than next to it.

    The provider is shared between the reward channel (through the gym wrapper)
    and the state channel (through the observation space's ``functs``), so a
    step's queries are evaluated once and reused.
    """
    act_attrs = DEFAULT_ACT_ATTRS
    safe_max_rho = None
    gymenv_extra: dict[str, Any] = {}
    if project_cfg is not None:
        act_attrs = tuple(project_cfg["actions"]["discrete_space"]["attr_to_keep"])
        if obs_attr_to_keep is None:
            obs_attr_to_keep = list(project_cfg["observation"]["attr_to_keep"])
        safe_max_rho = project_cfg["heuristics"].get("safe_max_rho")
        # the frozen config (v3+) pins how skipped steps' rewards are folded in;
        # read it rather than hardcode it so both sessions can never disagree
        if "reward_cumul" in project_cfg["heuristics"]:
            gymenv_extra["reward_cumul"] = project_cfg["heuristics"]["reward_cumul"]
        env_name = env_name or project_cfg["env"]["name"]
        if cfg.uses_provider and safe_max_rho is not None and cfg.risk_tau > safe_max_rho:
            warnings.warn(
                f"arm {cfg.arm}: risk_tau={cfg.risk_tau} > heuristics.safe_max_rho={safe_max_rho}. The wrapper "
                f"wakes the agent above {safe_max_rho} but nothing counts as at risk below {cfg.risk_tau}, so in "
                "that band the relevance mask leaves only do-nothing. Set risk_tau <= safe_max_rho.",
                stacklevel=2,
            )
    schema = GridSchema.from_env(env, env_name, act_attr_to_keep=act_attrs)
    provider = make_provider(schema, cfg, **provider_kwargs)

    base_attrs = list(obs_attr_to_keep if obs_attr_to_keep is not None else BASE_OBS_ATTRS)
    contract = contract or FairnessContract(obs_attr_to_keep=tuple(base_attrs), action_set_hash=schema.fingerprint)
    if check_fairness:
        fairness.check_all(provider, contract)

    return ArmSetup(
        cfg=cfg,
        schema=schema,
        provider=provider,
        gymenv_class=make_gymenv_class(cfg, provider, schema=schema, safe_max_rho=safe_max_rho, **gymenv_extra),
        obs_attr_to_keep=obs_space.extend_obs_attrs(base_attrs, provider),
        obs_space_kwargs={"functs": obs_space.build_functs(provider)},
        contract=contract,
    )


def train_arm(env: Any, setup: ArmSetup, iterations: int = 10_000, **kwargs: Any):
    """Train one arm through the vendored ``ppo_sb3.train``."""
    from grid2op_vkg.baselines.ppo_sb3 import train

    params = copy.deepcopy(kwargs)
    params.setdefault("name", setup.cfg.arm)
    params.update(setup.train_kwargs())
    return train(env, iterations=iterations, **params)


def evaluate_arm(env: Any, setup: ArmSetup, **kwargs: Any):
    """Evaluate one arm through the vendored ``ppo_sb3.evaluate``."""
    from grid2op_vkg.baselines.ppo_sb3 import evaluate

    params = copy.deepcopy(kwargs)
    params.setdefault("name", setup.cfg.arm)
    params.update(setup.evaluate_kwargs())
    return evaluate(env, **params)
