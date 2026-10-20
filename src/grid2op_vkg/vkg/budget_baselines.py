"""Equal-budget baselines without knowledge, for the scale experiment.

On case14 the whole action space fits in six budget-steps, so any search finds
the good move and knowledge is worth a few points. On 118 substations the space
is 72 958 actions and a 30-simulation budget sees 0.04 % of it per decision;
the question becomes *which* 30, and that is what the semantic index answers.
To show it, the semantic search needs two controls at the same budget:

``Random-Legal-Budget``     30 legal actions drawn uniformly at random each step
                            -- search with no knowledge at all
``Random-Semantic-Budget``  30 actions drawn uniformly from the semantic set
                            -- membership without ordering, i.e. the mask alone

Both inherit everything else from :class:`SemanticGreedyAgent`: the same
regime test, the same accept rule, the same free reconnection, the same
simulate() accounting. The only line that differs is which candidates get
simulated.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from grid2op_vkg.vkg.channels.mask import legality_mask
from grid2op_vkg.vkg.greedy import SemanticGreedyAgent


class RandomBudgetAgent(SemanticGreedyAgent):
    """Spend the budget on a uniform sample of ``pool`` instead of the semantic ordering."""

    def __init__(self, env: Any, project_cfg: dict, pool: str = "legal", **kw: Any) -> None:
        if pool not in ("legal", "semantic"):
            raise ValueError(f"pool must be 'legal' or 'semantic', got {pool!r}")
        super().__init__(env, project_cfg, **kw)  # the parent owns the rng and honours draw_seed
        if self.rewrite_dead or self.lookahead or self.memory:
            raise NotImplementedError(f"{self.name}: search-mechanics switches are only wired for greedy/hybrid agents")
        self.pool = pool

    def act(self, observation, reward, done=False):
        obs = observation
        self.provider.begin_step(obs)
        semantic = self.provider.action_mask()
        guidance = self.provider.search_guidance()
        self.provider.end_step()

        reco = np.flatnonzero(guidance.reconnectable)
        if reco.size:
            return self.action_space({"set_line_status": [(int(reco[0]), +1)]})

        legal = legality_mask(obs, self.schema)
        mask = legal & semantic if self.pool == "semantic" else legal
        mask[0] = False
        if not guidance.at_risk.any():
            # calm: the semantic agent only ever considers merge-backs here; give
            # the controls the same restriction so they are not penalised for
            # gambling on a calm grid
            mask &= guidance.recovery
        candidates = np.flatnonzero(mask)
        if candidates.size == 0:
            return self.action_space()
        order = self._permute(candidates, guidance)

        if not guidance.at_risk.any():
            best_idx, best_rho, spent = 0, np.inf, 0
            for idx in order:
                if spent >= self.budget:
                    break
                rho = self._simulate(obs, int(idx))
                spent += 1
                if rho < best_rho:
                    best_idx, best_rho = int(idx), rho
            if best_idx != 0 and best_rho < self.tau:
                self.n_merged += 1
                return self._g2op_actions[best_idx]
            return self.action_space()

        baseline = self._simulate(obs, 0)
        spent, best_idx, best_rho = 1, 0, baseline
        for idx in order:
            if spent >= self.budget:
                break
            rho = self._simulate(obs, int(idx))
            spent += 1
            if rho < best_rho:
                best_idx, best_rho = int(idx), rho
        if best_idx == 0:
            return self.action_space()
        if best_rho <= baseline - self.improvement or (baseline >= 1.0 and best_rho < 1.0):
            self.n_acted += 1
            if self.catalogue and guidance.remedial[best_idx]:
                self.n_played_from_catalogue += 1
            return self._g2op_actions[best_idx]
        return self.action_space()


def _register() -> None:
    from grid2op_vkg.registry import register_agent

    # The harness runs one seed and passes it positionally, so a kwarg named
    # "seed" collides before it reaches us. A stochastic baseline needs several
    # draws, so the override is called draw_seed:
    #   --agents "Random-Legal-Budget:s1" --agent-kwargs '{"Random-Legal-Budget:s1": {"draw_seed": 1}}'
    @register_agent("Random-Legal-Budget")
    def _legal(env, cfg, seed, **kw):
        return RandomBudgetAgent(env, cfg, pool="legal", name="Random-Legal-Budget", **kw)

    @register_agent("Random-Semantic-Budget")
    def _semantic(env, cfg, seed, **kw):
        return RandomBudgetAgent(env, cfg, pool="semantic", name="Random-Semantic-Budget", **kw)

    # the remedial-action catalogue on the equal-budget semantic searcher:
    # electrical membership + catalogue hits first, and the membership-only control
    @register_agent("Random-Semantic-Catalogue")
    def _semantic_catalogue(env, cfg, seed, **kw):
        return RandomBudgetAgent(env, cfg, pool="semantic", name="Random-Semantic-Catalogue", catalogue=True, **kw)

    @register_agent("Random-Semantic-Catalogue-Ranked")
    def _semantic_catalogue_ranked(env, cfg, seed, **kw):
        return RandomBudgetAgent(
            env,
            cfg,
            pool="semantic",
            name="Random-Semantic-Catalogue-Ranked",
            catalogue=True,
            catalogue_order="score",
            **kw,
        )

    @register_agent("Random-Semantic-Catalogue-Flat")
    def _semantic_catalogue_flat(env, cfg, seed, **kw):
        return RandomBudgetAgent(
            env,
            cfg,
            pool="semantic",
            name="Random-Semantic-Catalogue-Flat",
            catalogue=True,
            catalogue_order="flat",
            **kw,
        )


_register()
