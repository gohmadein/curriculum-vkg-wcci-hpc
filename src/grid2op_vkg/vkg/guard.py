"""Feasibility guard: do not play an injection action the simulator says ends the game.

The autopsy of the 60-chronic hybrid run found that 15 of OptimCVXPY's 21
blackouts are not blackouts: in safe mode it restores curtailment towards its
reference, the environment's automatic redispatch cannot balance the change, and
grid2op raises ``ImpossibleRedispatching`` and ends the episode on a calm grid
(rho 0.64-0.92, no line over its limit). ``obs.simulate`` reproduces the failure
one step ahead, so one simulation per acted step is enough to see it coming.

The guard is a fix to the *baseline's* behaviour, not knowledge from the graph,
and is therefore provided in two places so the table can separate the two:

``OptimCVXPY-Guarded``         the baseline plus the guard, no topology at all
``Hybrid-OptimFirst-Guarded``  the hybrid with the same guard on its injection lever

When the guard rejects an action it plays do-nothing for that step and clears
OptimCVXPY's flow prediction, exactly as OptimCVXPY does in its own do-nothing
mode; the optimiser re-solves from the next observation.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from grid2op.Agent import BaseAgent

from grid2op_vkg.vkg.hybrid import HybridAgent


def action_ends_game(obs, action) -> bool:
    """One simulation: does playing ``action`` next step end the episode or raise?"""
    try:
        _, _, done, info = obs.simulate(action)
    except Exception:  # noqa: BLE001  a simulator failure is as bad as a game over
        return True
    return bool(done) or bool(info.get("exception"))


def is_nothing(action) -> bool:
    return not action.as_dict()


class GuardedOptimCVXPY(BaseAgent):
    """OptimCVXPY whose action is simulated first; an action that ends the game is replaced by do-nothing."""

    def __init__(self, env: Any, rho_danger: float = 0.95, rho_safe: float = 0.85) -> None:
        super().__init__(env.action_space)
        from grid2op_vkg.baselines.optim_cvxpy import OptimCVXPY

        self.optim = OptimCVXPY(env.action_space, env, rho_danger=rho_danger, rho_safe=rho_safe)
        self.n_guarded = 0
        self.n_simulate = 0

    def act(self, observation, reward, done=False):
        act = self.optim.act(observation, reward, done)
        if is_nothing(act):
            return act
        self.n_simulate += 1
        if action_ends_game(observation, act):
            self.n_guarded += 1
            self.optim.flow_computed[:] = np.nan
            return self.action_space()
        return act


class GuardedHybridAgent(HybridAgent):
    """HybridAgent whose injection action passes through the same guard before being played.

    Topology actions are already simulated by the search (that is how they are
    chosen), so only the injection lever needs the extra check. In optim-first
    mode the escalation test already simulates OptimCVXPY's action; that result
    is reused so the guard costs nothing there.
    """

    def __init__(self, env: Any, project_cfg: dict, **kw: Any) -> None:
        super().__init__(env, project_cfg, **kw)
        self.n_guarded = 0

    def _guarded(self, obs, optim_act):
        if is_nothing(optim_act):
            return optim_act
        self.n_simulate += 1
        if action_ends_game(obs, optim_act):
            self.n_guarded += 1
            self._topology_played()  # clears the optimiser's flow prediction
            return self.action_space()
        return optim_act

    def act(self, observation, reward, done=False):
        obs = observation
        mask, guidance = self._perceive(obs)
        optim_act = self.optim.act(obs, reward, done)

        if not self._stressed(guidance):
            idx = self._search_recovery(obs, mask, guidance)
            if idx:
                act = self.action_space()
                act += optim_act
                act += self._candidate_action(idx)
                # the combined action is what will be played; guard that
                self.n_simulate += 1
                if action_ends_game(obs, act):
                    self.n_guarded += 1
                    self._topology_played()
                    return self.action_space()
                self.n_merged += 1
                self.n_combined += 1
                self._topology_played()
                return act
            out = self._guarded(obs, optim_act)
            if out is optim_act:
                self.n_optim_played += 1
            return out

        if self.mode == "optim_first":
            # one simulation serves both purposes: escalation test and guard
            try:
                sim_obs, _, sd, si = obs.simulate(optim_act)
                self.n_simulate += 1
                fails = bool(sd) or bool(si.get("exception"))
                rho_after = np.inf if fails else float(sim_obs.rho.max())
            except Exception:  # noqa: BLE001
                fails, rho_after = True, np.inf
            if not fails and rho_after < self._relief_target(guidance):
                self.n_optim_played += 1
                return optim_act
            if fails:
                self.n_guarded += 1
        idx = self._search_relief(obs, mask, guidance)
        if idx:
            self.n_acted += 1
            self.n_topo_played += 1
            self._topology_played()
            return self._action_for(idx)
        if self.mode == "optim_first":
            # nothing topological helped; the injection action is only playable if it does not end the game
            if fails:
                self._topology_played()
                return self.action_space()
            self.n_optim_played += 1
            return optim_act
        out = self._guarded(obs, optim_act)
        if out is optim_act:
            self.n_optim_played += 1
        return out


def _register() -> None:
    from grid2op_vkg.registry import register_agent

    @register_agent("OptimCVXPY-Guarded")
    def _g(env, cfg, seed, **kw):
        return GuardedOptimCVXPY(env, **kw)

    @register_agent("Hybrid-OptimFirst-Guarded")
    def _hg(env, cfg, seed, **kw):
        return GuardedHybridAgent(env, cfg, mode="optim_first", name="Hybrid-OptimFirst-Guarded", **kw)

    @register_agent("Hybrid-OptimFirst-Guarded-Catalogue")
    def _hg_catalogue(env, cfg, seed, **kw):
        return GuardedHybridAgent(
            env, cfg, mode="optim_first", name="Hybrid-OptimFirst-Guarded-Catalogue", catalogue=True, **kw
        )

    # the stack: guard (baseline repair) + VKG topology + catalogue (VKG) + dead-terminal rewrite (search
    # mechanics). The ablation on the April chronics showed the rewrite alone carries the rescues; lookahead
    # and memory are not part of it.
    @register_agent("Hybrid-OptimFirst-Guarded-Rewrite")
    def _hg_rewrite(env, cfg, seed, **kw):
        return GuardedHybridAgent(
            env, cfg, mode="optim_first", name="Hybrid-OptimFirst-Guarded-Rewrite", rewrite_dead=True, **kw
        )

    # catalogue tiered by "mix" (first 15 hits promoted, the rest left in the base order): the VKG
    # narrowing without displacing the order the hybrid already had
    @register_agent("Hybrid-OptimFirst-Guarded-Rewrite-Mix")
    def _hg_rewrite_mix(env, cfg, seed, **kw):
        return GuardedHybridAgent(
            env,
            cfg,
            mode="optim_first",
            name="Hybrid-OptimFirst-Guarded-Rewrite-Mix",
            catalogue=True,
            catalogue_order="mix",
            rewrite_dead=True,
            **kw,
        )

    @register_agent("Hybrid-OptimFirst-Guarded-Catalogue-Rewrite")
    def _hg_catalogue_rewrite(env, cfg, seed, **kw):
        return GuardedHybridAgent(
            env,
            cfg,
            mode="optim_first",
            name="Hybrid-OptimFirst-Guarded-Catalogue-Rewrite",
            catalogue=True,
            rewrite_dead=True,
            **kw,
        )


_register()
