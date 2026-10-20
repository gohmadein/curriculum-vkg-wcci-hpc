"""A rule-based agent built from the semantic mask -- no training, same budget.

ExpertAgent is the rule-based bar (59.9 % survival under the frozen config,
439 simulate() calls per episode). Its documented weakness is cooldown: once
it has split substation 4 it skips the locked substation and does not look
for an alternative. The knowledge graph already answers that question --
``legal_actions`` drops what is on cooldown, ``relevant_actions`` lists every
endpoint substation of a stressed line -- so the cheapest possible policy on
top of it is:

    when a line is at risk      simulate the relevant, legal candidates
                                (ranked, capped at the shared budget) and
                                play the one with the lowest resulting max rho
                                if it beats doing nothing
    when the grid is calm       merge a split substation back if simulation
                                says that stays calm; reconnect what can be
    otherwise                   do nothing

It exists for two reasons: it is a direct "VKG rules vs expert rules"
comparison that needs no GPU-hours, and its ``handcrafted`` twin must behave
identically -- the same construction as the learned arms.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from grid2op.Agent import BaseAgent

from grid2op_vkg.vkg.channels.mask import legality_mask, rewritable_actions
from grid2op_vkg.vkg.config import VkgConfig
from grid2op_vkg.vkg.providers import make_provider
from grid2op_vkg.vkg.schema import GridSchema


class SemanticGreedyAgent(BaseAgent):
    """Greedy one-step lookahead over the semantic action mask.

    Parameters
    ----------
    provider:
        ``"vkg"`` or ``"handcrafted"``.
    budget:
        Maximum ``obs.simulate()`` calls per step, *including* the do-nothing
        probe. Must not exceed the environment's ``MAX_SIMULATE_PER_STEP``.
    tau:
        Risk threshold; act when any line is at or above it. **1.0 is a tuned
        value, selected on val** (91.4 % against 85.5 % at 0.95) and it is the
        default precisely so that ``--agents VKG-Greedy`` with no kwargs is the
        tuned agent: on test, 0.95 scores 76.7 % and 1.0 scores 87.4 %, and a
        silently untuned default cost an afternoon chasing a regression that was
        not there. Acting only at the limit also leaves splits merged back for
        longer, which is what keeps substations available for the next overload.
    k:
        Relevance radius in hops.
    adjacency:
        ``substation`` (default here) or ``electrical``. ExpertAgent's cooldown
        work-around -- reconfigure substation 1 while substation 4 is locked --
        is only in the relevant set under substation adjacency: after the
        split, line 4 is no longer electrically adjacent to line 17.
    improvement:
        A candidate must lower the simulated max rho by at least this much
        below the do-nothing probe to be played.
    max_split_subs:
        Do not open a further substation once this many are already split;
        every split is a 3-step lock and a step away from the reference
        topology, and the merge-back has to be affordable later.
    rewrite_dead, lookahead, lookahead_target, memory, memory_ttl:
        Search mechanics, not knowledge -- each one flag, all off by default,
        and with all of them off the agent is bit-identical to the one
        without them. ``rewrite_dead`` admits candidates that are illegal only
        because they touch a dead line's terminal, with that terminal left
        alone (the substations next to a tripped line become usable again);
        the grid2op action is built only when it is simulated or played.
        ``lookahead`` spends one simulation per step on doing nothing and
        decides the regime on the forecast; when only the forecast is
        stressed, an injection action counts as relief below
        ``lookahead_target``. ``memory`` re-simulates first the candidates
        that beat doing nothing at a stressed step in the last ``memory_ttl``
        steps. Ported from ``feat/schedule`` with identical decisions.
    catalogue, catalogue_path, catalogue_theta, catalogue_order:
        Load the remedial-action catalogue (``grid2op_vkg.vkg.remedial``) into
        the graph. Membership becomes electrical -- actions the catalogue says
        relieve a line at risk by at least ``catalogue_theta`` join the
        candidate set wherever their substation is -- and the search simulates
        those before the rest: ``catalogue_order="first"`` keeps the base
        order inside the hits, ``"score"`` sorts the hits by catalogued relief
        (largest first), ``"flat"`` leaves the order untouched (membership
        only). Off, the agent is bit-identical to the one before the catalogue
        existed.
    """

    def __init__(
        self,
        env: Any,
        project_cfg: dict,
        provider: str = "vkg",
        budget: int = 30,
        tau: float = 1.0,
        k: int = 1,
        adjacency: str = "substation",
        improvement: float = 0.02,
        max_split_subs: int = 2,
        name: str = "VKG-Greedy",
        order: str = "ranked",
        draw_seed: int | None = None,
        catalogue: bool = False,
        catalogue_path: str | None = None,
        catalogue_theta: float = 0.02,
        catalogue_order: str = "first",
        catalogue_mix: int = 15,
        rewrite_dead: bool = False,
        lookahead: bool = False,
        lookahead_target: float = 0.95,
        memory: bool = False,
        memory_ttl: int = 6,
    ) -> None:
        super().__init__(env.action_space)
        if order not in ("ranked", "sampled"):
            raise ValueError(f"order must be 'ranked' or 'sampled', got {order!r}")
        if catalogue_order not in ("first", "score", "flat", "mix"):
            raise ValueError(f"catalogue_order must be 'first', 'score', 'flat' or 'mix', got {catalogue_order!r}")
        self.catalogue = bool(catalogue)
        self.catalogue_theta = float(catalogue_theta)
        self.catalogue_order = catalogue_order
        self.catalogue_mix = int(catalogue_mix)
        self.rewrite_dead = bool(rewrite_dead)
        self.lookahead = bool(lookahead)
        self.lookahead_target = float(lookahead_target)
        self.memory = bool(memory)
        self.memory_ttl = int(memory_ttl)
        self.order = order
        # The harness calls agent.seed(<its one seed>) after construction, so a
        # stochastic agent could never be run at several draws through it.
        # draw_seed, when given, wins over whatever seed() is later handed.
        self._draw_seed = draw_seed
        self._rng = np.random.default_rng(0 if draw_seed is None else draw_seed)
        self.project_cfg = project_cfg
        tau = float(tau)
        cap = int(project_cfg["env"]["parameters"].get("MAX_SIMULATE_PER_STEP", -1))
        if cap > 0 and budget > cap:
            raise ValueError(f"budget {budget} exceeds the environment's MAX_SIMULATE_PER_STEP={cap}")
        self.budget = int(budget)
        self.tau = tau
        self.improvement = float(improvement)
        # the debt cap is a shared threshold now: it parameterises
        # AffordableActions, so both providers answer with the same number
        self.max_split_subs = int(max_split_subs)
        self.name = name

        act_attrs = tuple(project_cfg["actions"]["discrete_space"]["attr_to_keep"])
        self.schema = GridSchema.from_env(env, project_cfg["env"]["name"], act_attr_to_keep=act_attrs)
        self.actions = self.schema.require_actions()
        if self.catalogue:
            from grid2op_vkg.vkg.remedial import RemedialCatalogue

            env_name, fp = project_cfg["env"]["name"], self.actions.fingerprint
            path = catalogue_path or RemedialCatalogue.default_path(env_name, fp)
            self.schema = self.schema.with_remedial(RemedialCatalogue.load(path, self.actions.n, self.schema.n_line))
        self.cfg = VkgConfig(
            provider=provider,
            mask=True,
            arm=name,
            risk_tau=tau,
            relevance_k=k,
            relevance_adjacency=adjacency,
            max_split_subs=self.max_split_subs,
            remedial_theta=self.catalogue_theta,
            remedial_membership=self.catalogue,
        )
        self.provider = make_provider(self.schema, self.cfg)
        self._g2op_actions = [self.actions.to_grid2op(i, env.action_space) for i in range(self.actions.n)]
        self.n_simulate = 0
        self.n_acted = 0
        #: provenance of the last topology action played (None if none / do-nothing)
        self.last_justification: dict[str, Any] | None = None
        self.n_merged = 0
        #: catalogue accounting: stressed searches that had catalogue hits, and relief actions played from them
        self.n_catalogue_steps = 0
        self.n_played_from_catalogue = 0
        #: search-mechanics state and accounting (all inert with the flags off)
        self._budget_left = self.budget
        self._t = 0
        self._next_rho: float | None = None
        self._rewritten: dict[int, np.ndarray] = {}
        self._rewritten_act: dict[int, Any] = {}
        self._vetted: dict[int, tuple[float, float, int]] = {}  # idx -> (rho, baseline, step)
        self.n_rewritten_played = 0
        self.n_memory_hits = 0
        self.n_lookahead_steps = 0

    # ------------------------------------------------------------------ #
    def _order(self, candidates: np.ndarray, g) -> np.ndarray:
        if self.order == "sampled":
            base = self._tier(self._order_sampled(candidates, g), g)
        else:
            base = self._tier(self._order_ranked(candidates, g), g)
        return self._remembered_first(base, candidates)

    def _remembered_first(self, base: np.ndarray, candidates: np.ndarray) -> np.ndarray:
        """Search memory: candidates that beat doing nothing recently come first, best first; a no-op when off."""
        if not self.memory or not self._vetted or candidates.size == 0:
            return base
        fresh = {
            i: rho
            for i, (rho, baseline, t) in self._vetted.items()
            if self._t - t <= self.memory_ttl and np.isfinite(rho) and rho < baseline
        }
        cset = set(int(i) for i in candidates)
        first = sorted((i for i in fresh if i in cset), key=lambda i: fresh[i])
        if not first:
            return base
        self.n_memory_hits += 1
        head = set(first)
        return np.asarray(first + [int(i) for i in base if int(i) not in head], dtype=int)

    def _permute(self, candidates: np.ndarray, g) -> np.ndarray:
        """A uniform permutation (the knowledge-free controls' order), catalogue hits first when asked."""
        return self._tier(self._rng.permutation(candidates), g)

    def _tier(self, order: np.ndarray, g) -> np.ndarray:
        """Promote catalogue hits inside the base order; a no-op unless the catalogue is on.

        ``first``: every hit before the rest (both parts keep their order).
        ``score``: the hits sorted by catalogued relief, then the rest.
        ``mix``: only the first ``catalogue_mix`` hits are promoted, in base
        order; the remaining candidates stay exactly where the base order put
        them, hits included. ``mix`` exists because ``first`` displaces: on
        chronic 443 the plain hybrid's rescuing action (a plain semantic
        candidate, not a catalogue hit) fell out of the 30-simulation window
        once every hit was promoted, and the agent then died where it used to
        survive. ``flat``: no promotion at all (membership only).
        """
        if not self.catalogue or self.catalogue_order == "flat" or order.size == 0:
            return order
        hit_pos = np.flatnonzero(g.remedial[order])
        if hit_pos.size == 0:
            return order
        self.n_catalogue_steps += 1
        if self.catalogue_order == "mix":
            head = hit_pos[: self.catalogue_mix]
        elif self.catalogue_order == "score":
            head = hit_pos[np.argsort(-g.remedial_score[order[hit_pos]], kind="stable")]
        else:
            head = hit_pos
        if head.size == order.size:
            return order[head]
        keep = np.ones(order.size, dtype=bool)
        keep[head] = False
        return np.concatenate([order[head], order[keep]])

    def _order_sampled(self, candidates: np.ndarray, g) -> np.ndarray:
        """Same priors as the ranking, used as sampling weights instead of a hard order.

        On wcci_2022 the deterministic ranking scored no better than a uniform
        draw from the semantic set (7/6/7 paired): it probes the *same* 30
        candidates every time the state repeats, so a good action ranked 31st
        is never seen, while a draw explores. This keeps what the ranking knew
        -- priority substations first, cheap reconfigurations first, no single
        substation swallowing the budget -- as a distribution:

            w(a) = (2 if a's substation is a priority substation else 1)
                   / (1 + disruption(a))
                   / |candidates at a's substation|

        and orders the candidates by a Gumbel-top-k draw from w, i.e. a
        weighted permutation without replacement.
        """
        sub_of = self.actions.sub[candidates]
        _, inv, counts = np.unique(sub_of, return_inverse=True, return_counts=True)
        w = (np.where(g.priority_subs[sub_of], 2.0, 1.0) / (1.0 + g.disruption[candidates])) / counts[inv]
        keys = np.log(w) + self._rng.gumbel(size=w.size)
        return candidates[np.argsort(-keys)]

    def _order_ranked(self, candidates: np.ndarray, g) -> np.ndarray:
        """Spread the budget across substations, cheapest configuration first.

        Pure arithmetic over :class:`SearchGuidance`: group by substation, put
        the substations the graph flagged as priority first, then the ones with
        fewest candidates, and interleave round-robin so a substation with 57
        configurations cannot swallow the budget before another is probed at
        all. That is what used to hide ExpertAgent's cooldown fallback.

        Nothing here mentions rho, topology or cooldown -- those arrived as
        ``g.priority_subs`` and ``g.disruption``. On a 118-substation grid this
        ordering is not a refinement, it is the only reason a 30-simulation
        budget sees anything useful.
        """
        sub_of = self.actions.sub
        by_sub: dict[int, list[int]] = {}
        for i in candidates:
            by_sub.setdefault(int(sub_of[i]), []).append(int(i))
        for lst in by_sub.values():
            lst.sort(key=lambda i: (float(g.disruption[i]), i))
        subs = sorted(by_sub, key=lambda sub: (0 if g.priority_subs[sub] else 1, len(by_sub[sub]), sub))
        out: list[int] = []
        depth = 0
        while len(out) < len(candidates):
            for sub in subs:
                lst = by_sub[sub]
                if depth < len(lst):
                    out.append(lst[depth])
            depth += 1
        return np.asarray(out, dtype=int)

    def _simulate(self, obs, idx: int) -> float:
        """Max rho after ``idx`` next step; +inf if the simulation fails or ends the game."""
        return self._simulate_action(obs, self._candidate_action(idx))

    def _simulate_action(self, obs, action) -> float:
        """Max rho after an arbitrary action next step; +inf on failure. Counts in n_simulate."""
        self.n_simulate += 1
        sim_obs, _, done, info = obs.simulate(action)
        if done or info.get("is_illegal") or info.get("is_ambiguous") or info.get("exception"):
            return np.inf
        return float(sim_obs.rho.max())

    def _candidate_action(self, idx: int):
        """The grid2op action for candidate ``idx``: its rewritten form (built on first use) when it had to be."""
        vec = self._rewritten.get(idx)
        if vec is None:
            return self._g2op_actions[idx]
        act = self._rewritten_act.get(idx)
        if act is None:
            act = self.action_space({"set_bus": [(int(t), int(vec[t])) for t in np.flatnonzero(vec)]})
            self._rewritten_act[idx] = act
        return act

    def _action_for(self, idx: int):
        """The grid2op action to play for a relief search result."""
        if idx in self._rewritten:
            self.n_rewritten_played += 1
        return self._candidate_action(idx)

    def _stressed(self, guidance) -> bool:
        """Regime: a line at risk now or -- with ``lookahead`` -- a forecast at or above tau."""
        if guidance.at_risk.any():
            return True
        return bool(self.lookahead and self._next_rho is not None and self._next_rho >= self.tau)

    def _relief_target(self, guidance) -> float:
        """Max rho an injection action must reach to count as relief in this regime."""
        return self.tau if guidance.at_risk.any() else self.lookahead_target

    def _baseline(self, obs) -> tuple[float, int]:
        """Do-nothing max rho next step and what it cost now (0 when the lookahead already paid for it)."""
        if self._next_rho is not None:
            return self._next_rho, 0
        return self._simulate(obs, 0), 1

    # ------------------------------------------------------------------ #
    def _perceive(self, obs):
        """Ask the provider once; return (semantic mask, guidance)."""
        self.provider.begin_step(obs)
        semantic = self.provider.action_mask()
        guidance = self.provider.search_guidance()
        self.provider.end_step()
        if guidance is None:  # a provider with no budget layer cannot drive this agent
            raise RuntimeError(f"{self.name}: provider {self.provider.name!r} offers no search guidance")
        self._budget_left = self.budget
        self._t += 1
        self._next_rho = None
        if self.lookahead:
            self._next_rho = self._simulate_action(obs, self.action_space())
            self._budget_left -= 1
            if self._next_rho >= self.tau and not guidance.at_risk.any():
                self.n_lookahead_steps += 1
        legal = legality_mask(obs, self.schema)
        self._rewritten, self._rewritten_act = {}, {}
        if self.rewrite_dead:
            self._rewritten = rewritable_actions(obs, self.schema, legal)
            if self._rewritten:
                legal[np.fromiter(self._rewritten, dtype=np.int64)] = True
        mask = legal & semantic
        mask[0] = False  # do-nothing is the fallback, not a candidate
        return mask, guidance

    def _search_recovery(self, obs, mask, guidance) -> int:
        """Calm grid: best merge-back that keeps the grid calm, or 0 for none.

        Restoring the reference topology is worth a little rho. A split left in
        place is a substation unavailable when the next overload comes, and
        there will be one -- ExpertAgent merges back at rho ~0.89 for exactly
        that reason. So the test is "does it stay calm", not "does it beat
        doing nothing".
        """
        candidates = np.flatnonzero(mask & guidance.recovery)
        if candidates.size == 0:
            return 0
        best_idx, best_rho, spent = 0, np.inf, 0
        for idx in self._order(candidates, guidance):
            if spent >= self._budget_left:
                break
            rho = self._simulate(obs, int(idx))
            spent += 1
            if rho < best_rho:
                best_idx, best_rho = int(idx), rho
        return best_idx if (best_idx != 0 and best_rho < self.tau) else 0

    def _search_relief(self, obs, mask, guidance) -> int:
        """Stressed grid: best affordable candidate if it clearly beats doing nothing, else 0.

        Records ``last_justification`` for the action it returns.
        """
        candidates = np.flatnonzero(mask & guidance.affordable)
        if candidates.size == 0:
            return 0
        baseline, spent = self._baseline(obs)
        best_idx, best_rho = 0, baseline
        for idx in self._order(candidates, guidance):
            if spent >= self._budget_left:
                break
            rho = self._simulate(obs, int(idx))
            spent += 1
            if self.memory:
                self._vetted[int(idx)] = (rho, baseline, self._t)
            if rho < best_rho:
                best_idx, best_rho = int(idx), rho
        if best_idx == 0:
            return 0
        # play it if it clearly beats doing nothing, or if doing nothing overloads and it does not
        if not (best_rho <= baseline - self.improvement or (baseline >= 1.0 and best_rho < 1.0)):
            return 0
        if self.catalogue and guidance.remedial[best_idx]:
            self.n_played_from_catalogue += 1
        witness = self.provider.justify(best_idx)
        self.last_justification = {
            "action": best_idx,
            "substation": int(self.actions.sub[best_idx]),
            "risk_lines": np.flatnonzero(guidance.at_risk).tolist(),
            "from_catalogue": bool(self.catalogue and guidance.remedial[best_idx]),
            "catalogued_relief": float(guidance.remedial_score[best_idx]),
            "witness_pairs": [(int(r), int(s)) for r, s in np.argwhere(witness)] if witness is not None else None,
            "baseline_rho": float(baseline),
            "chosen_rho": float(best_rho),
            "n_simulated": int(spent),
            "n_candidates": int(candidates.size),
        }
        return best_idx

    def act(self, observation, reward, done=False):
        """Search inside whatever the graph offers, and nothing else.

        Everything grid-specific arrives through the provider: the candidate
        mask, and the :class:`SearchGuidance` fields. What is left here is
        search policy -- reconnect for free, decide a regime, order, spend the
        budget, accept or decline. It reads no observation attribute, so a
        change in the data layer reaches it only through the mappings.
        """
        obs = observation
        mask, guidance = self._perceive(obs)

        # Reconnection first: it costs no budget and is always right. It cannot
        # come from the candidate set -- the frozen action space is set_bus only
        # -- so it is built directly, from the query rather than from the
        # observation.
        reco = np.flatnonzero(guidance.reconnectable)
        if reco.size:
            return self.action_space({"set_line_status": [(int(reco[0]), +1)]})

        if not self._stressed(guidance):
            idx = self._search_recovery(obs, mask, guidance)
            if idx:
                self.n_merged += 1
                return self._candidate_action(idx)
            return self.action_space()

        idx = self._search_relief(obs, mask, guidance)
        if idx:
            self.n_acted += 1
            return self._action_for(idx)
        return self.action_space()

    def seed(self, seed):
        self._rng = np.random.default_rng(seed if self._draw_seed is None else self._draw_seed)
        return super().seed(seed)


# --------------------------------------------------------------------------- #
# registry entries
# --------------------------------------------------------------------------- #
def _register() -> None:
    from grid2op_vkg.registry import register_agent

    @register_agent("VKG-Greedy")
    def _vkg(env, cfg, seed, **kw):
        return SemanticGreedyAgent(env, cfg, provider="vkg", name="VKG-Greedy", **kw)

    @register_agent("Handcrafted-Greedy")
    def _hand(env, cfg, seed, **kw):
        return SemanticGreedyAgent(env, cfg, provider="handcrafted", name="Handcrafted-Greedy", **kw)

    @register_agent("VKG-Greedy-Sampled")
    def _vkg_sampled(env, cfg, seed, **kw):
        return SemanticGreedyAgent(env, cfg, provider="vkg", name="VKG-Greedy-Sampled", order="sampled", **kw)

    @register_agent("VKG-Greedy-Catalogue")
    def _vkg_catalogue(env, cfg, seed, **kw):
        return SemanticGreedyAgent(env, cfg, provider="vkg", name="VKG-Greedy-Catalogue", catalogue=True, **kw)


_register()
