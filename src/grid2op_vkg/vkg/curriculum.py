"""Pretrained CurriculumAgent (senior, IEEE14) with the matched VKG / terminal-rewriting add-ons.

Four cells share one agent and differ only in two switches:

A  original: the senior ranks its 125 actions and simulates them in order until one
   brings the maximum loading to <= 0.95, within the 40-call budget per step
B  +VKG: the senior's loop may use ``budget - reserve`` calls; if it has not reached
   0.95 by then, the reserve is spent on VKG candidates (relevant to a line at risk,
   non-islanding, legal, random order per ``draw_seed``) with the senior's own
   acceptance rule. A spends those same calls on its next-ranked actions, so B - A
   compares candidates at equal budget. ``candidate_source="legal"`` is the random
   control: every legal set_bus action instead of the VKG pool.
C  +TR: a candidate that would set a bus on a disconnected line's terminal (which the
   senior's own legality rule rejects) is rewritten to leave that terminal alone
D  +VKG+TR
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np

from grid2op_vkg.baselines.curriculum_agent import MyAgent
from grid2op_vkg.baselines.curriculum_agent.utilities import simulate_action
from grid2op_vkg.registry import register_agent
from grid2op_vkg.vkg.backends.compiled import CompiledBackend
from grid2op_vkg.vkg.channels.mask import legality_mask
from grid2op_vkg.vkg.queries.action import IslandingActions, RelevantActions
from grid2op_vkg.vkg.schema import GridSchema
from grid2op_vkg.vkg.state import GridState

ROOT = Path(__file__).resolve().parents[3]
ACTIONS = ROOT / "src/grid2op_vkg/baselines/curriculum_agent/model_IEEE14/actions/actions.npy"
POLICY = ROOT / "data/curriculum/senior_ieee14_policy.npz"
POLICY_SHA256 = "eb2618469e7f84da15a542d209f9b86cf0268c116bb35409fe6654a256d44d52"
ADOPTED = -2  # truthy marker for a supplementary action (not one of the senior's indices)


class CurriculumFactorialAgent(MyAgent):
    def __init__(
        self,
        env,
        cfg,
        *,
        semantic_supplement=False,
        rewrite_dead=False,
        candidate_source="vkg",
        reserve=10,
        budget=40,
        draw_seed=0,
        audit_label="agent",
        policy_path=POLICY,
        policy_sha256=POLICY_SHA256,
        actions_path=ACTIONS,
        restrict_to_policy_actions=False,
    ):
        if hashlib.sha256(Path(policy_path).read_bytes()).hexdigest() != policy_sha256:
            raise ValueError(f"{policy_path}: not the expected exported CurriculumAgent policy")
        if candidate_source not in ("vkg", "legal"):
            raise ValueError(f"candidate_source must be 'vkg' or 'legal', got {candidate_source!r}")
        if not 0 < reserve < budget:
            raise ValueError("reserve must leave the senior part of the budget")
        super().__init__(env.action_space, policy_path, actions_path, budget=int(budget))
        self.env = env
        self.semantic_supplement = bool(semantic_supplement)
        self.rewrite_dead = bool(rewrite_dead)
        self.candidate_source = candidate_source
        self.reserve = int(reserve)
        self.draw_seed = int(draw_seed)
        self.rng = np.random.default_rng(draw_seed)
        self.audit_label = audit_label
        self._audit_dir = os.environ.get("VKG_CURRICULUM_AUDIT_DIR")
        self.schema = GridSchema.from_env(
            env, env.name, act_attr_to_keep=() if restrict_to_policy_actions else ("set_bus",))
        if restrict_to_policy_actions:
            # Same387train-derived actions in the policy and the semantic pool.
            from dataclasses import replace
            from grid2op_vkg.vkg.actions import CandidateActionSet
            buses = np.vstack([np.zeros(self.schema.dim_topo, np.int8),
                *[env.action_space.from_vect(v).set_bus.astype(np.int8) for v in self.actions]])
            n = len(buses)
            kind = np.r_[np.int8(0), np.ones(n-1, np.int8)]
            subs = np.full(n, -1, np.int32)
            for i in range(1, n):
                touched = np.unique(self.schema.topo_sub[buses[i] != 0])
                if len(touched) != 1: raise ValueError("Nonunitary policy action")
                subs[i] = touched[0]
            line = np.full(n, -1, np.int32); status = np.zeros(n, np.int8)
            digest = hashlib.sha256()
            for array in [kind, subs, line, status, buses]:
                digest.update(np.ascontiguousarray(array).tobytes())
                array.flags.writeable = False
            digest.update(b"set_bus")
            actions = CandidateActionSet(n, ("set_bus",), kind, subs, line, status, buses, digest.hexdigest()[:16])
            self.schema = replace(self.schema, actions=actions)
            object.__setattr__(self.schema, "_fingerprint", self.schema._compute_fingerprint())
            self.schema.validate()
        self._backend = None
        self._rewritten = set()
        self._trace = hashlib.sha256()
        self.simulate_calls = self.peak_calls = self.refused_calls = self.stressed_steps = 0
        self.search_exhaustions = self.supplement_triggers = self.supplement_calls = self.supplement_adopted = 0
        self.rewritten_candidates = self.rewritten_played = 0

    def _search_budget(self):
        return self.budget - self.reserve if self.semantic_supplement else self.budget

    def _candidate_vect(self, action_vect, obs):
        if not self.rewrite_dead:
            return action_vect
        sb = np.asarray(self.action_space.from_vect(action_vect).set_bus).copy()
        dead = np.zeros(self.schema.dim_topo, bool)
        dead[self.schema.line_terminals[~np.asarray(obs.line_status, bool)].ravel()] = True
        live = ~dead & (np.asarray(obs.topo_vect) > 0)
        if not (sb[dead] > 0).any() or not ((sb != 0) & (sb != obs.topo_vect) & live).any():
            return action_vect
        sb[dead] = 0
        vect = self.action_space({"set_bus": sb}).to_vect()
        self._rewritten.add(vect.tobytes())
        self.rewritten_candidates += 1
        return vect

    def _pool(self, obs):
        a = self.schema.require_actions()
        if self.candidate_source == "vkg":
            if self._backend is None:
                self._relevance = RelevantActions(tau=self.best_action_threshold, k=1, adjacency="substation")
                self._island = IslandingActions()
                self._backend = CompiledBackend(self.schema, [self._relevance, self._island])
            self._backend.begin_step(GridState.from_obs(self.schema, obs))
            semantic = self._backend.ask(self._relevance.name) & ~self._backend.ask(self._island.name)
        else:
            semantic = np.ones(a.n, dtype=bool)
        legal = legality_mask(obs, self.schema)
        if self.rewrite_dead:
            legal |= a.is_set_bus & (np.asarray(obs.time_before_cooldown_sub)[np.maximum(a.sub, 0)] == 0)
        live = np.asarray(obs.topo_vect) > 0
        for index in self.rng.permutation(np.flatnonzero(semantic & legal & a.is_set_bus)):
            vect = self._candidate_vect(a.to_grid2op(int(index), self.action_space).to_vect(), obs)
            sb = np.asarray(self.action_space.from_vect(vect).set_bus)
            if ((sb != 0) & (sb != obs.topo_vect) & live).any():
                yield vect

    def _after_search(self, obs, idx_chosen, chosen_vect, min_rho):
        idx_chosen, chosen_vect = self._supplement(obs, idx_chosen, chosen_vect, min_rho)
        self._chosen = chosen_vect if idx_chosen else None
        return idx_chosen, chosen_vect

    def _supplement(self, obs, idx_chosen, chosen_vect, min_rho):
        self.stressed_steps += 1
        if self.sim.calls >= self._search_budget():
            self.search_exhaustions += 1
        if not self.semantic_supplement or min_rho <= self.best_action_threshold:
            return idx_chosen, chosen_vect
        if self.sim.calls < self._search_budget():
            return idx_chosen, chosen_vect  # the senior's loop ended for another reason
        self.supplement_triggers += 1
        before = self.sim.calls
        for vect in self._pool(obs):
            if self.sim.calls >= self.budget:
                break
            rho_max, valid = simulate_action(self.action_space, obs, vect, self.check_overload)
            if valid and rho_max < min_rho:
                min_rho, idx_chosen, chosen_vect = rho_max, ADOPTED, vect
                if rho_max <= self.best_action_threshold:
                    break
        self.supplement_calls += self.sim.calls - before
        self.supplement_adopted += int(idx_chosen == ADOPTED)
        return idx_chosen, chosen_vect

    def act_with_id(self, observation):
        self._chosen = None
        action, idx = super().act_with_id(observation)
        if self._chosen is not None and self._chosen.tobytes() in self._rewritten:
            chosen_sb = np.asarray(self.action_space.from_vect(self._chosen).set_bus)
            self.rewritten_played += int(np.array_equal(np.asarray(action.set_bus), chosen_sb))
        self.simulate_calls += self.sim.calls
        self.refused_calls += self.sim.refused
        self.peak_calls = max(self.peak_calls, self.sim.calls)
        self._trace.update(action.to_vect().tobytes())
        self._audit(observation)
        return action, idx

    def _audit(self, obs):
        if not self._audit_dir:
            return
        record = {
            "label": self.audit_label,
            "last_step": int(obs.current_step),
            "semantic_supplement": self.semantic_supplement,
            "rewrite_dead": self.rewrite_dead,
            "candidate_source": self.candidate_source,
            "draw_seed": self.draw_seed,
            "reserve": self.reserve,
            "budget": self.budget,
            "simulate_calls": self.simulate_calls,
            "max_simulate_per_step": self.peak_calls,
            "refused_calls": self.refused_calls,
            "stressed_steps": self.stressed_steps,
            "search_exhaustions": self.search_exhaustions,
            "supplement_triggers": self.supplement_triggers,
            "supplement_calls": self.supplement_calls,
            "supplement_adopted": self.supplement_adopted,
            "rewritten_candidates": self.rewritten_candidates,
            "rewritten_played": self.rewritten_played,
            "action_trace_sha256": self._trace.hexdigest(),
        }
        chronic = Path(self.env.chronics_handler.get_id()).name
        path = Path(self._audit_dir) / f"{self.audit_label}__{chronic}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(json.dumps(record))
        temporary.replace(path)


@register_agent("Curriculum-Factorial")
def build(env, cfg, seed, **kwargs):
    return CurriculumFactorialAgent(env, cfg, **kwargs)


@register_agent("Curriculum-WCCI")
def build_wcci(env, cfg, seed, *, model_dir, cell="A", **kwargs):
    folder = Path(model_dir)
    meta = json.loads((folder / "run_meta.json").read_text())
    if meta["status"] != "complete" or meta["config_sha"] != cfg["_sha256"]:
        raise ValueError("Incomplete or incompatible Curriculum senior")
    switches = {"A": {}, "B": {"semantic_supplement": True},
                "C": {"rewrite_dead": True},
                "D": {"semantic_supplement": True, "rewrite_dead": True}}
    if cell not in switches: raise ValueError(cell)
    return CurriculumFactorialAgent(
        env, cfg, policy_path=folder/"policy.npz", policy_sha256=meta["policy_sha256"],
        actions_path=folder/"actions.npy", restrict_to_policy_actions=True, draw_seed=int(meta["seed"]),
        audit_label=f"{cell}_s{meta['seed']}", **switches[cell], **kwargs)
