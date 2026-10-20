"""Expert-first, residual-budget VKG search. No filtering of expert candidates."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation

from grid2op_vkg.registry import register_agent
from grid2op_vkg.vkg.backends.compiled import CompiledBackend
from grid2op_vkg.vkg.channels.mask import legality_mask
from grid2op_vkg.vkg.expert import AdaptedExpertSimulation, FactorialExpertAgent, action_key
from grid2op_vkg.vkg.queries.action import IslandingActions, RelevantActions
from grid2op_vkg.vkg.schema import GridSchema
from grid2op_vkg.vkg.state import GridState


class ScoredExpertSimulation(AdaptedExpertSimulation):
    def compute_new_network_changes(self, ranked_combinations):
        scores, actions = super().compute_new_network_changes(ranked_combinations)
        for (_, row), action in zip(scores.iterrows(), actions, strict=not scores.empty):
            record = (float(row["Topology simulated score"]), float(row["Efficacity"]), self)
            previous = self.adapter._scores.get(action_key(action))
            if previous is None or record[0] > previous[0]:
                self.adapter._scores[action_key(action)] = record
        return scores, actions


class SupplementExpertAgent(FactorialExpertAgent):
    def __init__(self, env, cfg, *, enabled=True, draw_seed=0, candidate_source="vkg", **kwargs):
        super().__init__(env, cfg, semantic_filter=False, **kwargs)
        if candidate_source not in ("vkg", "legal"):
            raise ValueError(f"candidate_source must be 'vkg' or 'legal', got {candidate_source!r}")
        # "legal" is the matched control: same trigger, budget, scoring and draw
        # order, but candidates are every legal set_bus action (no VKG relevance
        # or islanding filter) -- isolates what the graph adds beyond the budget.
        self.candidate_source = candidate_source
        self.enabled = bool(enabled)
        self.rng = np.random.default_rng(draw_seed)
        self.draw_seed = int(draw_seed)
        self.supplement_triggers = self.supplement_calls = self.supplement_adopted = 0
        self.satisfied_unchanged = self.extra_rewritten_played = 0
        self._pool_schema = self._backend = None
        self._scores = {}
        self._last_simulator = None
        original = self._expert_act

        def expert_then_supplement(agent, obs, reward, done):
            agent._scores = {}
            agent._last_simulator = None
            before_subs = agent.sub_2nodes.copy()
            before_lines = agent.lines_disconnected.copy()
            selected = original(agent, obs, reward, done)
            if not agent.enabled or not np.any(obs.rho >= 1.0) or agent._calls >= agent.budget:
                return selected
            result = agent._supplement(obs, selected)
            if action_key(result) != action_key(selected):
                # Discard bookkeeping for a replaced, never-executed expert action.
                agent.sub_2nodes = before_subs
                agent.lines_disconnected = before_lines
                sb = np.asarray(result.set_bus)
                for sub in np.unique(agent.schema.topo_sub[np.flatnonzero(sb)]):
                    positions = np.flatnonzero(agent.schema.topo_sub == sub)
                    topo = np.asarray(obs.topo_vect)[positions].copy()
                    change = sb[positions] != 0
                    topo[change] = sb[positions][change]
                    if np.any(topo == 2):
                        agent.sub_2nodes.add(int(sub))
                    else:
                        agent.sub_2nodes.discard(int(sub))
            return result

        self._expert_act = expert_then_supplement

    def _simulator(self, *args, **kwargs):
        simulator = ScoredExpertSimulation(*args, adapter=self, **kwargs)
        self._last_simulator = simulator
        return simulator

    def _prepare_pool(self):
        if self._pool_schema is None:
            self._pool_schema = GridSchema.from_env(self._env, self._env.name, act_attr_to_keep=("set_bus",))
            self._relevance = RelevantActions(tau=1.0, k=1, adjacency="substation")
            self._island = IslandingActions()
            self._backend = CompiledBackend(self._pool_schema, [self._relevance, self._island])

    def _candidates(self, obs):
        self._prepare_pool()
        schema = self._pool_schema
        a = schema.require_actions()
        self._backend.begin_step(GridState.from_obs(schema, obs))
        if self.candidate_source == "vkg":
            semantic = self._backend.ask(self._relevance.name) & ~self._backend.ask(self._island.name)
        else:
            semantic = np.ones(a.n, dtype=bool)
        legal = legality_mask(obs, schema)
        if self.rewrite_dead:
            # Conversion is lazy; don't construct thousands of Grid2Op actions.
            legal |= a.is_set_bus & (obs.time_before_cooldown_sub[np.maximum(a.sub, 0)] == 0)
        indices = np.flatnonzero(semantic & legal & a.is_set_bus)
        for index in self.rng.permutation(indices):
            action = a.to_grid2op(int(index), self.action_space)
            action = self.rewrite_action(action, obs)
            sb = np.asarray(action.set_bus)
            live = np.asarray(obs.topo_vect) > 0
            if not ((sb != 0) & (sb != obs.topo_vect) & live).any():
                continue
            if action_key(action) not in self._simulated:
                yield action, int(a.sub[index])

    @staticmethod
    def _better(score, best):
        if not np.isfinite(score[1]) or score[0] <= 0:
            return False
        return score[0] > best[0] or (score[0] == best[0] and score[1] > best[1])

    def _score_action(self, obs, action, simulator, sub):
        # Reuse the exact expert scoring routine and its simulation horizon.
        virtual, _, done, info = obs.simulate(action, time_step=max(simulator.simu_step - obs.current_step, 0))
        self.supplement_calls += 1
        if done or info.get("is_illegal") or info.get("is_ambiguous") or info.get("exception"):
            return (-1.0, -np.inf)
        cfg = np.asarray(action.set_bus)[self.schema.topo_sub == sub] if sub >= 0 else np.array([0, 0])
        score = Grid2opSimulation.compute_one_network_change_score_data(
            simulator, obs, virtual, done, info, cfg, sub, cfg, cfg, -1
        )
        fields = dict(zip(simulator.create_end_result_empty_dataframe().columns, score, strict=True))
        return float(fields["Topology simulated score"]), float(fields["Efficacity"])

    def _supplement(self, obs, selected):
        known = self._scores.get(action_key(selected))
        if known is not None and known[0] >= 3:
            self.satisfied_unchanged += 1
            return selected
        simulator = known[2] if known is not None else self._last_simulator
        if simulator is None:
            return selected
        self.supplement_triggers += 1
        if known is None:
            # Do not assume the returned no-op is worthless: score it fairly.
            best = self._score_action(obs, selected, simulator, -1)
            if best[0] >= 3:
                self.satisfied_unchanged += 1
                return selected
        else:
            best = known[:2]
        if not np.isfinite(best[1]):
            best = (best[0], -np.inf)
        result = selected
        if self._calls >= self.budget:
            return result
        for action, sub in self._candidates(obs):
            if self._calls >= self.budget:
                break
            score = self._score_action(obs, action, simulator, sub)
            if self._better(score, best):
                best, result = score, action
                if score[0] >= 4:
                    break
        if action_key(result) != action_key(selected):
            self.supplement_adopted += 1
            self.extra_rewritten_played += int(action_key(result) in self._rewritten)
        return result

    def _audit(self, obs):
        super()._audit(obs)
        self._last_audit.update(
            supplement_enabled=self.enabled,
            candidate_source=self.candidate_source,
            draw_seed=self.draw_seed,
            supplement_triggers=self.supplement_triggers,
            supplement_calls=self.supplement_calls,
            supplement_adopted=self.supplement_adopted,
            satisfied_unchanged=self.satisfied_unchanged,
            extra_rewritten_played=self.extra_rewritten_played,
        )
        if self._audit_dir:
            chronic = Path(self._env.chronics_handler.get_id()).name
            path = Path(self._audit_dir) / f"{self.audit_label}__{chronic}.json"
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(self._last_audit))
            temporary.replace(path)


@register_agent("Expert-Supplement")
def build(env, cfg, seed, **kwargs):
    return SupplementExpertAgent(env, cfg, **kwargs)


@register_agent("Expert-Supplement-Random")
def build_random(env, cfg, seed, **kwargs):
    return SupplementExpertAgent(env, cfg, candidate_source="legal", **kwargs)
