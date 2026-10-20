"""Passive validation-only expert traces; no new actions or simulator calls."""

from __future__ import annotations

import hashlib
import inspect
import json
import os
from pathlib import Path

import numpy as np

from grid2op_vkg.registry import register_agent
from grid2op_vkg.vkg.backends.compiled import CompiledBackend
from grid2op_vkg.vkg.expert import AdaptedExpertSimulation, FactorialExpertAgent, action_key, candidate_schema
from grid2op_vkg.vkg.queries.action import IslandingActions, RelevantActions
from grid2op_vkg.vkg.state import GridState


def key(action):
    return hashlib.sha256(action_key(action)).hexdigest()


def masks(schema, actions, obs):
    view = candidate_schema(schema, actions)
    relevant = RelevantActions(tau=1.0, k=1, adjacency="substation")
    island = IslandingActions()
    backend = CompiledBackend(view, [relevant, island])
    try:
        backend.begin_step(GridState.from_obs(view, obs))
        return backend.ask(relevant.name).copy(), backend.ask(island.name).copy()
    finally:
        backend.close()


class DiagnosticSimulation(AdaptedExpertSimulation):
    def compute_new_network_changes(self, ranked_combinations):
        # Includes partially scored batches when the original budget stop fires.
        self._records = []
        frame = inspect.currentframe().f_back
        callers = []
        while frame is not None and len(callers) < 8:
            callers.append(frame.f_code.co_name)
            frame = frame.f_back
        self._branch = next(
            (n for n in callers if n in ("bonus_action_IEEE14", "try_out_reference_topologies")),
            "expert_search",
        )
        try:
            return super().compute_new_network_changes(ranked_combinations)
        finally:
            if self._records:
                actions = [a for a, _ in self._records]
                relevant, island = masks(self.adapter.schema, actions, self.obs)
                for (action, record), rel, isl in zip(self._records, relevant, island, strict=True):
                    sb = np.asarray(action.set_bus)
                    dead = np.zeros(self.adapter.schema.dim_topo, bool)
                    dead[self.adapter.schema.line_terminals[~self.obs.line_status].ravel()] = True
                    live = ~dead & (self.obs.topo_vect > 0)
                    record.update(
                        relevant=bool(rel),
                        structural_reject=bool(isl),
                        keep=bool(rel and not isl),
                        rewrite_possible=bool(
                            (sb[dead] > 0).any() and ((sb != 0) & (sb != self.obs.topo_vect) & live).any()
                        ),
                    )
                    self.adapter._candidates.append(record)

    def get_action_from_topo(self, substation_id, new_conf, obs):
        action = super().get_action_from_topo(substation_id, new_conf, obs)
        self._candidate_action = action
        return action

    def compute_one_network_change_score_data(self, *args, **kwargs):
        score = super().compute_one_network_change_score_data(*args, **kwargs)
        obs, virtual, done, info, _, node, alpha_topo, _, _ = args
        action = (
            self.action_space({"set_line_status": [(int(alpha_topo[0]), -1)]})
            if len(alpha_topo) == 1
            else self._candidate_action
        )
        values = dict(zip(self.create_end_result_empty_dataframe().columns, score, strict=True))
        efficacy = float(values["Efficacity"])
        self._records.append(
            (
                action,
                dict(
                    key=key(action),
                    branch=self._branch,
                    node=int(node),
                    score=float(values["Topology simulated score"]),
                    efficacy=efficacy if np.isfinite(efficacy) else None,
                    done=bool(done),
                    illegal=bool(info.get("is_illegal", False)),
                    rho_after=float(np.max(virtual.rho)),
                    rho_before=float(np.max(obs.rho)),
                    set_bus=[[int(i), int(action.set_bus[i])] for i in np.flatnonzero(action.set_bus)],
                    set_line=[[int(i), int(action.set_line_status[i])] for i in np.flatnonzero(action.set_line_status)],
                ),
            )
        )
        return score


class DiagnosticExpert(FactorialExpertAgent):
    def _simulator(self, *args, **kwargs):
        return DiagnosticSimulation(*args, adapter=self, **kwargs)

    def act(self, observation, reward, done=False):
        self._candidates = []
        state_hash = hashlib.sha256(
            b"".join(
                np.asarray(getattr(observation, attr)).tobytes()
                for attr in ("topo_vect", "line_status", "rho", "prod_p", "load_p", "time_before_cooldown_sub")
            )
        ).hexdigest()
        before = self.budget_exhaustions
        action = super().act(observation, reward, done)
        overloaded = bool(np.any(observation.rho >= 1.0))
        if overloaded or self._candidates or action.as_dict():
            selected = key(action)
            matches = [r for r in self._candidates if r["key"] == selected]
            scored_nodes = {r["node"] for r in self._candidates if r["set_bus"]}
            available = set(
                np.flatnonzero((observation.time_before_cooldown_sub == 0) & (self.schema.sub_info >= 3)).tolist()
            )
            record = dict(
                step=int(observation.current_step),
                state_hash=state_hash,
                selected=selected,
                noop=not bool(action.as_dict()),
                overloaded=overloaded,
                calls=self._calls,
                remaining=max(0, self.budget - self._calls),
                budget_stop=self.budget_exhaustions > before,
                selected_scored=bool(matches),
                selected_rejected=any(not r["keep"] for r in matches),
                best_observed_score=max((r["score"] for r in self._candidates), default=None),
                untested_unlocked_subs=sorted(available - scored_nodes),
                candidates=self._candidates,
            )
            chronic = Path(self._env.chronics_handler.get_id()).name
            path = Path(os.environ["VKG_EXPERT_DIAGNOSTIC_DIR"]) / f"{self.audit_label}__{chronic}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with path.open("a") as stream:
                stream.write(json.dumps(record, allow_nan=False) + "\n")
        return action


@register_agent("Expert-Diagnostic")
def build(env, cfg, seed, **kwargs):
    return DiagnosticExpert(env, cfg, **kwargs)
