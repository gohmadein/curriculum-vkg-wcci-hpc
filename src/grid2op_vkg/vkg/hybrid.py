"""OptimCVXPY plus the semantic topology search: two levers, one agent.

On wcci_2022 the two strongest agents fail on different chronics. OptimCVXPY
(redispatch + curtailment + storage, never topology) reaches 80.6 % but dies at
step 400 on `2050-04-04_6` and `2050-04-18_4`; the semantic topology search
(set_bus only) reaches 57.4 % and runs those two to 693 and 1 588 steps. This
agent puts the two together so the question "does the knowledge graph add
anything on top of the strongest baseline?" has a direct answer: baseline =
OptimCVXPY alone, treatment = OptimCVXPY + the semantic search.

Two escalation orders, because it is not obvious which lever to pull first:

``topo_first``   when a line is at risk, play a topology action if the graph
                 offers one that relieves it (the greedy agent's own rule);
                 otherwise play OptimCVXPY's injection action
``optim_first``  when a line is at risk, simulate OptimCVXPY's injection action
                 first (one simulation); if that already brings the grid under
                 the limit, play it; otherwise escalate to topology

In both, OptimCVXPY runs every step so its internal flow-error correction stays
consistent, and whenever a topology action is played its flow prediction is
invalidated (``flow_computed[:] = nan``), which is exactly what it does itself
in its do-nothing mode. On a calm grid OptimCVXPY's recovery action (back to
zero redispatch, reconnections) is combined with the semantic merge-back when
one is available -- one grid2op action, both levers. Reconnection is left
entirely to OptimCVXPY, so that on any chronic where no topology action fires
the hybrid's trajectory is bit-identical to the baseline's.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from grid2op_vkg.vkg.greedy import SemanticGreedyAgent


class HybridAgent(SemanticGreedyAgent):
    def __init__(
        self,
        env: Any,
        project_cfg: dict,
        mode: str = "topo_first",
        rho_danger: float = 0.95,
        rho_safe: float = 0.85,
        **kw: Any,
    ) -> None:
        if mode not in ("topo_first", "optim_first"):
            raise ValueError(f"mode must be 'topo_first' or 'optim_first', got {mode!r}")
        super().__init__(env, project_cfg, **kw)
        from grid2op_vkg.baselines.optim_cvxpy import OptimCVXPY

        # same construction as the registry's OptimCVXPY baseline, so the
        # injection lever is the baseline's, not a retuned one
        self.optim = OptimCVXPY(env.action_space, env, rho_danger=rho_danger, rho_safe=rho_safe)
        self.mode = mode
        self.n_topo_played = 0
        self.n_optim_played = 0
        self.n_combined = 0

    def _topology_played(self) -> None:
        # our topology change invalidates OptimCVXPY's flow prediction for this
        # step; clearing it is what it does itself when it plays nothing
        self.optim.flow_computed[:] = np.nan

    def act(self, observation, reward, done=False):
        obs = observation
        mask, guidance = self._perceive(obs)
        optim_act = self.optim.act(obs, reward, done)

        # Reconnection is OptimCVXPY's: it reconnects in safe mode, inside the same
        # optimum. The greedy agent's eager rule (reconnect the moment a line
        # leaves cooldown, whatever rho) was inherited here at first and made
        # the hybrid differ from the baseline on chronics where no topology
        # action was ever played -- 2 of 5 differences on the 40-chronic
        # confirmation set. With it gone, the only thing this agent does that
        # OptimCVXPY does not is escalate to topology.
        if not self._stressed(guidance):
            idx = self._search_recovery(obs, mask, guidance)
            if idx:
                act = self.action_space()
                act += optim_act
                act += self._candidate_action(idx)
                self.n_merged += 1
                self.n_combined += 1
                self._topology_played()
                return act
            self.n_optim_played += 1
            return optim_act

        if self.mode == "optim_first":
            if self._simulate_action(obs, optim_act) < self._relief_target(guidance):
                self.n_optim_played += 1
                return optim_act
        idx = self._search_relief(obs, mask, guidance)
        if idx:
            self.n_acted += 1
            self.n_topo_played += 1
            self._topology_played()
            return self._action_for(idx)
        self.n_optim_played += 1
        return optim_act


def _register() -> None:
    from grid2op_vkg.registry import register_agent

    @register_agent("Hybrid-TopoFirst")
    def _topo(env, cfg, seed, **kw):
        return HybridAgent(env, cfg, mode="topo_first", name="Hybrid-TopoFirst", **kw)

    @register_agent("Hybrid-OptimFirst")
    def _optim(env, cfg, seed, **kw):
        return HybridAgent(env, cfg, mode="optim_first", name="Hybrid-OptimFirst", **kw)

    # feat/schedule's Hybrid-OptimFirst-Plus, ported: the port check reproduces its steps
    @register_agent("Hybrid-OptimFirst-Plus")
    def _optim_plus(env, cfg, seed, **kw):
        return HybridAgent(
            env,
            cfg,
            mode="optim_first",
            name="Hybrid-OptimFirst-Plus",
            rewrite_dead=True,
            memory=True,
            lookahead=True,
            **kw,
        )

    @register_agent("Hybrid-OptimFirst-Catalogue")
    def _optim_catalogue(env, cfg, seed, **kw):
        return HybridAgent(env, cfg, mode="optim_first", name="Hybrid-OptimFirst-Catalogue", catalogue=True, **kw)


_register()
