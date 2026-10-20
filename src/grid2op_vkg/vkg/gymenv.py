"""The gym environment every ablation arm trains in.

It subclasses ``GymEnvWithHeuristics`` for two reasons. First, the reconnection
heuristic is then identical across arms, so nothing in the comparison depends on
who reconnects lines. Second, ``ppo_sb3.evaluate`` only builds a wrapper at all
when ``issubclass(gymenv_class, GymEnvWithHeuristics)`` -- a plain ``GymEnv``
subclass would be silently ignored at evaluation time.

Reward shaping lives here. State injection does not; see
:mod:`grid2op_vkg.vkg.obs_space` for why.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from grid2op.Action import BaseAction

from grid2op_vkg.baselines.utils.gymenv_custom import GymEnvWithHeuristics
from grid2op_vkg.vkg.channels.mask import compose_mask, legality_mask
from grid2op_vkg.vkg.config import VkgConfig
from grid2op_vkg.vkg.providers.base import FeedbackProvider
from grid2op_vkg.vkg.providers.null import NullProvider


class SemanticGymEnv(GymEnvWithHeuristics):
    """Reconnection heuristic + optional potential-based reward shaping.

    Parameters
    ----------
    cfg:
        The arm's configuration. With the default ``Raw`` config this class
        behaves exactly like ``GymEnvWithReco``.
    provider:
        Supplies Phi(s). The same object is normally also wired into the
        observation space, so the two channels share one set of query results
        per step rather than computing them twice.
    safe_max_rho:
        Below this the grid is treated as safe and the heuristic plays do-nothing
        instead of waking the agent. ``None`` disables that behaviour: skipping
        steps makes the process semi-Markov (see the gamma**k correction below).
    reward_cumul:
        How the rewards of the environment steps the heuristic plays on the
        agent's behalf are folded into the agent's reward. ``"sum"`` -- and it
        has to be; the frozen config (v3, ``heuristics.reward_cumul``) says so
        and ``runner.build_arm`` passes it through. The upstream defaults
        (``"init"`` in GymEnvWithRecoWithDN, ``"last"`` in the base class) keep
        a single step's reward, so an agent whose action keeps the grid calm
        for 500 steps is paid exactly as much as one whose action leads to a
        blackout 4 steps later: survival is not rewarded, and every learned arm
        trained that way collapsed to do-nothing.
    """

    def __init__(
        self,
        env_init,
        *args,
        cfg: VkgConfig | None = None,
        provider: FeedbackProvider | None = None,
        schema=None,
        safe_max_rho: float | None = None,
        reward_cumul: str = "sum",
        **kwargs,
    ) -> None:
        super().__init__(env_init, *args, reward_cumul=reward_cumul, **kwargs)
        self.cfg = cfg if cfg is not None else VkgConfig()
        self.provider = provider if provider is not None else NullProvider()
        self.schema = schema
        self._safe_max_rho = safe_max_rho
        self.mask_sizes: list[int] = []

        self._prev_potential: float | None = None
        #: grid2op steps consumed by the last gym step, for the gamma**k term
        self._steps_taken: int = 1
        self.shaping_total: float = 0.0
        self.n_shaped_steps: int = 0

    # ------------------------------------------------------------------ #
    # heuristics -- identical for every arm
    # ------------------------------------------------------------------ #
    def heuristic_actions(self, g2op_obs, reward, done, info) -> list[BaseAction]:
        to_reco = (g2op_obs.time_before_cooldown_line == 0) & (~g2op_obs.line_status)
        if np.any(to_reco):
            return [
                self.init_env.action_space({"set_line_status": [(int(line_id), +1)]})
                for line_id in np.flatnonzero(to_reco)
            ]
        if self._safe_max_rho is not None and g2op_obs.rho.max() <= self._safe_max_rho:
            return [self.init_env.action_space()]
        return []

    def apply_heuristics_actions(self, g2op_obs, reward, done, info):
        """Same as the parent, but counts how many grid2op steps were consumed.

        The count matters: with heuristics in the loop one gym step can span
        several environment steps, so the process the agent sees is semi-Markov
        and the shaping term needs ``gamma**k`` rather than ``gamma`` to stay
        policy-invariant.
        """
        before = self.init_env.nb_time_step
        out = super().apply_heuristics_actions(g2op_obs, reward, done, info)
        self._steps_taken = max(1, int(self.init_env.nb_time_step) - int(before) + 1)
        return out

    # ------------------------------------------------------------------ #
    # reward shaping
    # ------------------------------------------------------------------ #
    def _potential(self, g2op_obs) -> float | None:
        if not self.cfg.reward:
            return None
        self.provider.begin_step(g2op_obs)
        phi = self.provider.potential()
        self.provider.end_step()
        return phi

    def _shape(self, reward: float, g2op_obs, done: bool) -> float:
        """``r' = r + lambda * (gamma**k * Phi(s') - Phi(s))``.

        On termination Phi(s') is taken to be 0: the terminal state has no
        successor, and letting the potential float there would make blackouts
        look better or worse than they are.
        """
        if not self.cfg.reward:
            return reward
        phi_next = 0.0 if done else (self._potential(g2op_obs) or 0.0)
        phi_prev = self._prev_potential
        self._prev_potential = None if done else phi_next
        if phi_prev is None:
            return reward
        gamma_k = self.cfg.gamma**self._steps_taken
        bonus = self.cfg.shaping_lambda * (gamma_k * phi_next - phi_prev)
        self.shaping_total += bonus
        self.n_shaped_steps += 1
        return reward + bonus

    # ------------------------------------------------------------------ #
    def step(self, gym_action):
        self._steps_taken = 1
        g2op_act = self.fix_action(self.action_space.from_gym(gym_action), self._previous_act)
        g2op_obs, reward, done, info = self.init_env.step(g2op_act)
        if not done:
            g2op_obs, reward, done, info = self.apply_heuristics_actions(g2op_obs, reward, done, info)

        reward = self._shape(float(reward), g2op_obs, done)
        self._previous_act = g2op_obs
        gym_obs = self.observation_space.to_gym(g2op_obs)

        info = dict(info)
        info["vkg_steps_taken"] = self._steps_taken
        if self.cfg.aux:
            # the aux channel may be the only one switched on, so make sure the
            # provider has seen this observation (begin_step is idempotent)
            self.provider.begin_step(g2op_obs)
            aux = self.provider.aux_targets()
            self.provider.end_step()
            if aux is not None:
                info["vkg_aux"] = aux

        if getattr(type(self), "_gymnasium", False):
            return gym_obs, float(reward), done, False, info
        return gym_obs, float(reward), done, info

    def reset(self, *, seed=None, return_info=False, options=None):
        self._prev_potential = None
        self._steps_taken = 1
        out = super().reset(seed=seed, return_info=return_info, options=options)
        gym_obs = out[0] if isinstance(out, tuple) else out
        if self.cfg.reward:
            self._prev_potential = self._potential(self.init_env.get_obs())
        return out if isinstance(out, tuple) else gym_obs

    # ------------------------------------------------------------------ #
    # action masking -- MaskablePPO calls env.action_masks() before predict
    # ------------------------------------------------------------------ #
    def mask_for(self, obs) -> np.ndarray:
        """Legality for every arm; AND the provider's semantic filter if it has one."""
        if self.schema is None or self.schema.actions is None:
            raise RuntimeError("masking needs a schema with a candidate action set")
        legal = legality_mask(obs, self.schema) if self.cfg.legality_mask else np.ones(self.schema.actions.n, bool)
        semantic = None
        if self.cfg.mask:
            self.provider.begin_step(obs)
            semantic = self.provider.action_mask()
            self.provider.end_step()
        mask = compose_mask(legal, semantic)
        self.mask_sizes.append(int(mask.sum()))
        return mask

    def action_masks(self) -> np.ndarray:
        """MaskablePPO's hook: mask for the state the agent is about to act in."""
        obs = self._previous_act if self._previous_act is not None else self.init_env.get_obs()
        return self.mask_for(obs)

    def summary(self) -> dict[str, float]:
        """Shaping bookkeeping, for checking lambda actually did something."""
        out = {
            "shaping_total": self.shaping_total,
            "n_shaped_steps": self.n_shaped_steps,
            "shaping_mean": self.shaping_total / self.n_shaped_steps if self.n_shaped_steps else 0.0,
        }
        if self.mask_sizes:
            xs = np.asarray(self.mask_sizes)
            out.update(mask_size_mean=float(xs.mean()), mask_size_min=int(xs.min()), mask_size_max=int(xs.max()))
        return out


def make_gymenv_class(cfg: VkgConfig, provider: FeedbackProvider, **extra: Any):
    """Bind cfg and provider into a zero-argument class.

    ``ppo_sb3.train`` instantiates ``gymenv_class(env, **gymenv_kwargs)``, and
    ``gymenv_kwargs`` is documented as not being saved or restored -- so the
    only reliable way to get the arm's configuration into the evaluation-time
    wrapper is to bake it into the class itself.
    """

    class _BoundSemanticGymEnv(SemanticGymEnv):
        def __init__(self, env_init, *args, **kwargs):
            kwargs.setdefault("cfg", cfg)
            kwargs.setdefault("provider", provider)
            for key, val in extra.items():
                kwargs.setdefault(key, val)
            super().__init__(env_init, *args, **kwargs)

    _BoundSemanticGymEnv.__name__ = f"SemanticGymEnv_{cfg.arm.replace('-', '_')}"
    _BoundSemanticGymEnv.__qualname__ = _BoundSemanticGymEnv.__name__
    return _BoundSemanticGymEnv
