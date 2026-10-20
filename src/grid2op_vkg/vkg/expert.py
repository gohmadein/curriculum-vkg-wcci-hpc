"""ExpertAgent factorial adapter: retain the upstream decision/scoring pipeline.

Only its topology-candidate conversion and pre-simulation membership are
adapted. The original Python act code is rebound to an instance-local simulator
factory; no third-party module globals or evaluation code are patched.
All four cells share a 40-call budget and an explicit no-op on exhaustion.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
from pathlib import Path
from types import FunctionType

import numpy as np
from alphaDeesp.core.grid2op.Grid2opSimulation import Grid2opSimulation

from grid2op_vkg.baselines.expert_agent import ExpertAgent
from grid2op_vkg.registry import register_agent
from grid2op_vkg.vkg.actions import ActKind, CandidateActionSet
from grid2op_vkg.vkg.backends.compiled import CompiledBackend
from grid2op_vkg.vkg.queries.action import IslandingActions, RelevantActions
from grid2op_vkg.vkg.schema import GridSchema
from grid2op_vkg.vkg.state import GridState


class ExpertBudgetExhausted(RuntimeError):
    """The next call would exceed the shared per-decision budget."""


def action_key(action):
    return np.asarray(action.to_vect()).tobytes()


def candidate_schema(schema, actions):
    """Expose only this expert-generated batch as the existing VKG action view."""
    n = len(actions)
    kind = np.zeros(n, np.int8)
    sub = np.full(n, -1, np.int32)
    line = np.full(n, -1, np.int32)
    status = np.zeros(n, np.int8)
    buses = np.zeros((n, schema.dim_topo), np.int8)
    for i, action in enumerate(actions):
        sb, ls = np.asarray(action.set_bus), np.asarray(action.set_line_status)
        terms, lines = np.flatnonzero(sb), np.flatnonzero(ls)
        if terms.size:
            subs = np.unique(schema.topo_sub[terms])
            if subs.size != 1 or lines.size or (sb[terms] < 1).any():
                raise ValueError("Expected a single-substation expert topology candidate")
            kind[i], sub[i], buses[i] = ActKind.SET_BUS, subs[0], sb
        elif lines.size:
            if lines.size != 1:
                raise ValueError("Expected a single-line expert candidate")
            kind[i], line[i], status[i] = ActKind.LINE_STATUS, lines[0], ls[lines[0]]
    arrays = (kind, sub, line, status, buses)
    digest = hashlib.sha256(b"".join(a.tobytes() for a in arrays)).hexdigest()
    for arr in arrays:
        arr.flags.writeable = False
    candidates = CandidateActionSet(n, ("expert_batch",), *arrays, digest)
    return dataclasses.replace(schema, actions=candidates)


class AdaptedExpertSimulation(Grid2opSimulation):
    def __init__(self, *args, adapter, **kwargs):
        self.adapter = adapter
        super().__init__(*args, **kwargs)

    def get_action_from_topo(self, substation_id, new_conf, obs):
        original = super().get_action_from_topo(substation_id, new_conf, obs)
        return self.adapter.rewrite_action(original, obs)

    def compute_new_network_changes(self, ranked_combinations):
        if not self.adapter.semantic_filter:
            return super().compute_new_network_changes(ranked_combinations)
        # Use raw candidates for membership in B and D alike. Rewriting occurs
        # during upstream conversion, before both simulation and score recording.
        actions = []
        for frame in ranked_combinations:
            for _, row in frame.iterrows():
                topology = np.asarray(row["topology"])
                if topology.size == 1:
                    action = self.action_space({"set_line_status": [(int(topology[0]), -1)]})
                else:
                    action = Grid2opSimulation.get_action_from_topo(self, int(row["node"]), topology + 1, self.obs)
                actions.append(action)
        if actions:
            keep = self.adapter.membership(actions, self.obs)
            self.adapter.candidates_seen += len(actions)
            self.adapter.candidates_removed += int((~keep).sum())
            filtered, offset = [], 0
            for frame in ranked_combinations:
                filtered.append(frame.iloc[np.flatnonzero(keep[offset : offset + len(frame)])])
                offset += len(frame)
            ranked_combinations = filtered
        # Keep original row labels/order, per-node limits and score-selection
        # rules. Rejected rows do not consume simulation slots.
        return super().compute_new_network_changes(ranked_combinations)


class FactorialExpertAgent(ExpertAgent):
    def __init__(
        self,
        env,
        cfg,
        *,
        semantic_filter=False,
        rewrite_dead=False,
        budget=40,
        grid_name="IEEE14",
        audit_label="expert",
        reference=False,
    ):
        super().__init__(env.action_space, env.observation_space, "Expert-Factorial", gridName=grid_name)
        cap = int(cfg["env"]["parameters"]["MAX_SIMULATE_PER_STEP"])
        if budget < 1 or (cap >= 0 and budget > cap):
            raise ValueError("Expert budget must be positive and within the environment cap")
        if reference and (semantic_filter or rewrite_dead):
            raise ValueError("The upstream reference cannot enable treatments")
        self.semantic_filter, self.rewrite_dead = bool(semantic_filter), bool(rewrite_dead)
        self.budget, self.reference = int(budget), bool(reference)
        self.schema = GridSchema.from_env(env, cfg["env"]["name"], act_attr_to_keep=())
        self._env, self.audit_label = env, audit_label
        self._audit_dir = os.environ.get("VKG_EXPERT_AUDIT_DIR")
        self.simulate_calls = self.peak_calls = self.budget_exhaustions = self.probe_exceptions = 0
        self.candidates_seen = self.candidates_removed = self.rewritten_candidates = self.rewritten_played = 0
        self._trace = hashlib.sha256()
        self._calls = 0
        self._rewritten = set()
        self._simulated = set()
        self._last_audit = None
        self.exception_types = {}
        # Reuse upstream bytecode with an instance-scoped dependency injection.
        # This is isolated across agents even when a validation process has two.
        namespace = dict(ExpertAgent.act.__globals__)
        namespace["Grid2opSimulation"] = self._simulator
        self._expert_act = FunctionType(
            ExpertAgent.act.__code__, namespace, "expert_act", ExpertAgent.act.__defaults__, ExpertAgent.act.__closure__
        )

    def _simulator(self, *args, **kwargs):
        if not self.semantic_filter and not self.rewrite_dead:
            return Grid2opSimulation(*args, **kwargs)
        return AdaptedExpertSimulation(*args, adapter=self, **kwargs)

    def membership(self, actions, obs):
        schema = candidate_schema(self.schema, actions)
        relevant = RelevantActions(tau=1.0, k=1, adjacency="substation")
        island = IslandingActions()
        backend = CompiledBackend(schema, [relevant, island])
        try:
            backend.begin_step(GridState.from_obs(schema, obs))
            return backend.ask(relevant.name) & ~backend.ask(island.name)
        finally:
            backend.close()

    def rewrite_action(self, action, obs):
        if not self.rewrite_dead:
            return action
        vec = np.asarray(action.set_bus).copy()
        dead = np.zeros(self.schema.dim_topo, bool)
        dead[self.schema.line_terminals[~np.asarray(obs.line_status, bool)].ravel()] = True
        if not (vec[dead] > 0).any():
            return action
        live = ~dead & (np.asarray(obs.topo_vect) > 0)
        if not ((vec != 0) & (vec != obs.topo_vect) & live).any():
            return action
        vec[dead] = 0
        rewritten = self.action_space({"set_bus": vec})
        self._rewritten.add(action_key(rewritten))
        self.rewritten_candidates += 1
        return rewritten

    def reference_topology_sub_action(self, observation, sub_id):
        return self.rewrite_action(super().reference_topology_sub_action(observation, sub_id), observation)

    def _audit(self, observation):
        self.peak_calls = max(self.peak_calls, self._calls)
        self._last_audit = {
            "label": self.audit_label,
            "last_step": int(observation.current_step),
            "semantic_filter": self.semantic_filter,
            "rewrite_dead": self.rewrite_dead,
            "reference": self.reference,
            "budget": self.budget,
            "simulate_calls": self.simulate_calls,
            "max_simulate_per_step": self.peak_calls,
            "budget_exhaustions": self.budget_exhaustions,
            "probe_exceptions": self.probe_exceptions,
            "candidates_seen": self.candidates_seen,
            "candidates_removed": self.candidates_removed,
            "rewritten_candidates": self.rewritten_candidates,
            "rewritten_played": self.rewritten_played,
            "action_trace_sha256": self._trace.hexdigest(),
            "exception_types": self.exception_types,
        }
        if self._audit_dir:
            chronic = Path(self._env.chronics_handler.get_id()).name
            path = Path(self._audit_dir) / f"{self.audit_label}__{chronic}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            temp = path.with_suffix(".tmp")
            temp.write_text(json.dumps(self._last_audit))
            temp.replace(path)

    def act(self, observation, reward, done=False):
        self._calls = 0
        self._rewritten, self._simulated = set(), set()
        original_simulate = observation.simulate
        own_simulate = observation.__dict__.get("simulate")
        obs_env = observation._obs_env
        saved = (
            obs_env._no_overflow_disconnection,
            obs_env._parameters.NO_OVERFLOW_DISCONNECTION,
            obs_env._parameters.MAX_LINE_STATUS_CHANGED,
        )
        result = self.action_space()

        def counted(action, *args, **kwargs):
            if not self.reference and self._calls >= self.budget:
                self.budget_exhaustions += 1
                raise ExpertBudgetExhausted("Expert's shared simulation budget exhausted")
            self._calls += 1
            self.simulate_calls += 1
            self._simulated.add(action_key(action))
            return original_simulate(action, *args, **kwargs)

        observation.simulate = counted
        try:
            result = self._expert_act(self, observation, reward, done)
            if action_key(result) in self._rewritten:
                if action_key(result) not in self._simulated:
                    raise RuntimeError("Rewritten executed action was not simulated")
                self.rewritten_played += 1
            return result
        except ExpertBudgetExhausted:
            return result  # explicit no-op, same fallback as the shared harness
        except Exception as exc:
            self.probe_exceptions += 1
            name = type(exc).__name__
            self.exception_types[name] = self.exception_types.get(name, 0) + 1
            raise  # unexpected failures remain visible to the shared harness
        finally:
            if own_simulate is None:
                del observation.simulate
            else:
                observation.simulate = own_simulate
            # AlphaDeesp temporarily relaxes line-cut probe parameters. A budget
            # stop must not leave these altered for the next decision.
            if not self.reference:
                (
                    obs_env._no_overflow_disconnection,
                    obs_env._parameters.NO_OVERFLOW_DISCONNECTION,
                    obs_env._parameters.MAX_LINE_STATUS_CHANGED,
                ) = saved
            self._trace.update(action_key(result))
            self._audit(observation)


@register_agent("Expert-Factorial")
def _expert_factorial(env, cfg, seed, **kwargs):
    return FactorialExpertAgent(env, cfg, **kwargs)


@register_agent("Expert-Reference")
def _expert_reference(env, cfg, seed, **kwargs):
    return FactorialExpertAgent(env, cfg, reference=True, **kwargs)
