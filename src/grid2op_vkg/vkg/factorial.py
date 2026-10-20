"""Isolated 2x2 experiment: semantic candidate filtering x dead-terminal rewriting.

The shared heuristics remain identical in all four cells. "VKG off" means
candidate filtering off, not absence of every domain heuristic. The old agents
are unchanged. These agents impose a TOTAL simulate budget, including optimiser
checks, the do-nothing probe and the feasibility guard.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from grid2op_vkg.registry import register_agent
from grid2op_vkg.vkg.channels.mask import legality_mask
from grid2op_vkg.vkg.greedy import SemanticGreedyAgent
from grid2op_vkg.vkg.guard import GuardedHybridAgent, is_nothing


class FactorialMixin:
    def __init__(self, env, project_cfg, *, semantic_filter=True, audit_label="agent", **kw):
        if any(kw.get(k) for k in ("catalogue", "lookahead", "memory")):
            raise ValueError("The factorial experiment freezes catalogue, lookahead and memory OFF")
        super().__init__(env, project_cfg, **kw)
        if self.budget < 2:
            raise ValueError("The shared guard/search budget must be at least two")
        self.semantic_filter = bool(semantic_filter)
        self._audit_env = env
        self._audit_label = audit_label
        self._step_calls = 0
        self._peak_calls = 0
        self._budget_exhaustions = 0
        self._probe_exceptions = 0
        self._audit_dir = os.environ.get("VKG_FACTORIAL_AUDIT_DIR")

    def _perceive(self, obs):
        self._step_calls = 0
        mask, guidance = super()._perceive(obs)
        if not self.semantic_filter:
            mask = legality_mask(obs, self.schema)
            if self._rewritten:
                mask[np.fromiter(self._rewritten, dtype=np.int64)] = True
            mask[0] = False
        return mask, guidance

    def _probe(self, obs, action):
        if self._step_calls >= self.budget:
            self._budget_exhaustions += 1
            return None, True
        self._step_calls += 1
        self.n_simulate += 1
        try:
            simulated, _, done, info = obs.simulate(action)
        except Exception:
            self._probe_exceptions += 1
            raise  # the common harness records an agent error; do not hide it
        fails = bool(done or info.get("exception") or info.get("is_illegal") or info.get("is_ambiguous"))
        return simulated, fails

    def _simulate_action(self, obs, action):
        simulated, fails = self._probe(obs, action)
        return np.inf if fails else float(simulated.rho.max())

    def _guard_action(self, obs, action):
        if is_nothing(action):
            return action
        _, fails = self._probe(obs, action)
        if fails:
            self.n_guarded += 1
            self._topology_played()
            return self.action_space()
        return action

    def _audit(self, obs):
        self._peak_calls = max(self._peak_calls, self._step_calls)
        if not self._audit_dir:
            return
        chronic = Path(self._audit_env.chronics_handler.get_id()).name
        path = Path(self._audit_dir) / f"{self._audit_label}__{chronic}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "label": self._audit_label,
            "chronic": chronic,
            "last_step": int(obs.current_step),
            "semantic_filter": self.semantic_filter,
            "rewrite_dead": self.rewrite_dead,
            "budget": self.budget,
            "max_simulate_per_step": self._peak_calls,
            "simulate_calls": self.n_simulate,
            "budget_exhaustions": self._budget_exhaustions,
            "probe_exceptions": self._probe_exceptions,
            "rewritten_played": self.n_rewritten_played,
        }
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(payload))
        temporary.replace(path)

    def act(self, observation, reward, done=False):
        try:
            return self._act_factorial(observation, reward, done)
        finally:
            self._audit(observation)


class FactorialRandomAgent(FactorialMixin, SemanticGreedyAgent):
    """Uniform search, same recovery/reconnection heuristics in all four cells."""

    def _act_factorial(self, obs, reward, done=False):
        mask, guidance = self._perceive(obs)
        reco = np.flatnonzero(guidance.reconnectable)
        if reco.size:
            return self.action_space({"set_line_status": [(int(reco[0]), 1)]})
        calm = not guidance.at_risk.any()
        if calm:
            mask &= guidance.recovery
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return self.action_space()
        baseline = np.inf if calm else self._simulate(obs, 0)
        best_idx, best_rho = 0, baseline
        for idx in self._permute(candidates, guidance):
            if self._step_calls >= self.budget:
                break
            rho = self._simulate(obs, int(idx))
            if rho < best_rho:
                best_idx, best_rho = int(idx), rho
        accept = (
            best_rho < self.tau
            if calm
            else (best_rho <= baseline - self.improvement or (baseline >= 1.0 and best_rho < 1.0))
        )
        if best_idx and accept:
            if calm:
                self.n_merged += 1
            else:
                self.n_acted += 1
            return self._action_for(best_idx)
        return self.action_space()


class FactorialHybridAgent(FactorialMixin, GuardedHybridAgent):
    """Guarded OptimFirst with the same search and ranking in every cell."""

    def _act_factorial(self, obs, reward, done=False):
        mask, guidance = self._perceive(obs)
        optim_act = self.optim.act(obs, reward, done)
        if not self._stressed(guidance):
            # Reserve one call to validate the injection/recovery combination.
            self._budget_left = self.budget - 1
            idx = self._search_recovery(obs, mask, guidance)
            if idx:
                act = self.action_space()
                act += optim_act
                act += self._candidate_action(idx)
                checked = self._guard_action(obs, act)
                if checked is act:
                    self.n_merged += 1
                    self.n_combined += 1
                    self.n_rewritten_played += int(idx in self._rewritten)
                    self._topology_played()
                return checked
            self.n_optim_played += 1
            return self._guard_action(obs, optim_act)

        simulated, fails = self._probe(obs, optim_act)
        if not fails and float(simulated.rho.max()) < self._relief_target(guidance):
            self.n_optim_played += 1
            return optim_act
        if fails:
            self.n_guarded += 1
        self._budget_left = self.budget - self._step_calls
        idx = self._search_relief(obs, mask, guidance)
        if idx:
            self.n_acted += 1
            self.n_topo_played += 1
            self._topology_played()
            return self._action_for(idx)
        if fails:
            self._topology_played()
            return self.action_space()
        self.n_optim_played += 1
        return optim_act


@register_agent("Factorial-Random")
def _random(env, cfg, seed, **kw):
    return FactorialRandomAgent(env, cfg, name="Factorial-Random", **kw)


@register_agent("Factorial-Hybrid")
def _hybrid(env, cfg, seed, **kw):
    return FactorialHybridAgent(env, cfg, mode="optim_first", name="Factorial-Hybrid", **kw)
