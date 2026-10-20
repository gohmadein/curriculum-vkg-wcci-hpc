""" Port of ``curriculumagent.submission.my_agent.MyAgent`` (curriculumagent 1.1.2), the agent that the
l2rpn-baselines ``CurriculumAgent`` wraps. Only the Keras branch that the pretrained IEEE14 senior uses
is kept (functional model with a policy and a value head, full ``obs.to_vect()`` input, no scaler,
``topo=False``). Local changes are marked ``# LOCAL`` and listed in ``baselines/UPSTREAM.md``:

1. The TensorFlow model is replaced by the same MLP evaluated in numpy from exported weights.
2. The model input is the first 467 entries of ``obs.to_vect()`` (the observation layout the model
   was trained on; grid2op >= 1.11 appends ``gen_p_delta`` and ``timestep_protection_engaged``).
3. ``obs.simulate`` goes through a per-step proxy: identical calls within one step are served from a
   cache, and at most ``budget`` distinct calls are made; beyond it a call reports failure.
   The candidate loop stops when the budget is spent.
4. Three hooks with identity defaults, for the matched add-ons: ``_candidate_vect`` (rewrite a
   candidate before it is simulated), ``_search_budget`` (calls the senior's loop may use) and
   ``_after_search`` (may replace the chosen candidate).

Upstream behaviour is otherwise kept, including ``if idx_chosen:`` treating action 0 as "none".
"""

import logging

import grid2op
import numpy as np
from grid2op.Agent import BaseAgent

from grid2op_vkg.baselines.curriculum_agent.utilities import (
    find_best_line_to_reconnect,
    is_legal,
    simulate_action,
    split_action_and_return,
)


class _FailedSimulation:  # LOCAL
    rho = np.array([np.inf])


class _StepSimulator:  # LOCAL
    """Observation proxy: per-step cache and call budget for ``simulate``; everything else delegates."""

    def __init__(self, obs, budget):
        self._obs = obs
        self._budget = budget
        self._cache = {}
        self.calls = 0
        self.refused = 0

    def __getattr__(self, name):
        return getattr(self._obs, name)

    def simulate(self, action, *args, **kwargs):
        key = (action.to_vect().tobytes(), args, tuple(sorted(kwargs.items())))
        if key not in self._cache:
            if self._budget is not None and self.calls >= self._budget:
                self.refused += 1
                return _FailedSimulation(), 0.0, True, {"is_illegal": False, "is_ambiguous": False, "exception": []}
            self.calls += 1
            self._cache[key] = self._obs.simulate(action, *args, **kwargs)
        return self._cache[key]


class MyAgent(BaseAgent):
    def __init__(
            self,
            action_space,
            policy_path,
            action_space_path,
            best_action_threshold: float = 0.95,
            check_overload: bool = False,
            max_action_sim: int = 50,
            budget=None,
    ):
        BaseAgent.__init__(self, action_space=action_space)
        self.actions = np.load(str(action_space_path))
        weights = np.load(str(policy_path))  # LOCAL
        self.layers = [(weights[f"W{i}"], weights[f"b{i}"]) for i in range(4)]
        self.head = (weights["W_act"], weights["b_act"])
        self.n_inputs = self.layers[0][0].shape[0]
        self.check_overload = check_overload
        self.recovery_stack = []
        self.overflow_steps = 0
        self.next_actions = None
        self.best_action_threshold = best_action_threshold
        self.max_action_sim = max_action_sim
        self.budget = budget  # LOCAL
        self.topo = False
        self.sim = None

    def _simulator(self, observation):  # LOCAL
        self.sim = _StepSimulator(observation, self.budget)
        return self.sim

    def _exhausted(self):  # LOCAL
        return self.budget is not None and self.sim.calls >= self._search_budget()

    def _search_budget(self):  # LOCAL hook
        return self.budget

    def _candidate_vect(self, action_vect, observation):  # LOCAL hook
        return action_vect

    def _after_search(self, obs, idx_chosen, chosen_vect, min_rho):  # LOCAL hook
        return idx_chosen, chosen_vect

    def act_with_id(
            self, observation: grid2op.Observation.BaseObservation) -> grid2op.Action.BaseAction:
        observation = self._simulator(observation)  # LOCAL
        if self.next_actions is not None:
            # Try to do a step:
            try:
                next_action = next(self.next_actions)
                next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)
                if is_legal(next_action, observation):
                    return next_action, -1
            except StopIteration:
                self.next_actions = None

        if observation.rho.max() >= 1:
            self.overflow_steps += 1
        else:
            self.overflow_steps = 0

        # case: secure with low threshold
        if observation.rho.max() < self.best_action_threshold:  # fixed threshold
            default_action = self.action_space({})
            default_action = find_best_line_to_reconnect(obs=observation,
                                                         original_action=default_action)
            return default_action, -1

        # case: non-secure with high threshold
        min_rho = observation.rho.max()
        logging.info(
            f"{observation.get_time_stamp()}s, heavy load,"
            f" line-{observation.rho.argmax()}d load is {observation.rho.max()}"
        )

        idx_chosen = None
        chosen_vect = None  # LOCAL
        sorted_actions = self.__get_actions(obs=observation)[:self.max_action_sim]

        for k, idx in enumerate(sorted_actions):
            if self._exhausted():  # LOCAL
                break
            action_vect = self._candidate_vect(self.actions[idx, :], observation)  # LOCAL
            rho_max, valid_action = simulate_action(action_space=self.action_space, obs=observation,
                                                    action_vect=action_vect, check_overload=self.check_overload
                                                    )

            if not valid_action:
                continue

            if rho_max <= self.best_action_threshold:
                # For a very suitable candidate, we break the loop
                logging.info(f"take action {idx}, max-rho to {rho_max}," f" simulation times: {k + 1}")
                idx_chosen = idx
                chosen_vect = action_vect  # LOCAL
                min_rho = rho_max  # LOCAL (only read by _after_search)
                break

            if rho_max < min_rho:
                # If we have an action with a better rho than the current one, we still save it
                min_rho = rho_max
                idx_chosen = idx
                chosen_vect = action_vect  # LOCAL

        idx_chosen, chosen_vect = self._after_search(observation, idx_chosen, chosen_vect, min_rho)  # LOCAL

        if idx_chosen:
            self.next_actions = split_action_and_return(observation, self.action_space, chosen_vect)  # LOCAL
            next_action = next(self.next_actions)
            next_action = find_best_line_to_reconnect(obs=observation, original_action=next_action)

        else:
            next_action = find_best_line_to_reconnect(obs=observation, original_action=self.action_space({}))
            idx_chosen = -1

        return next_action, idx_chosen

    def act(self, observation: grid2op.Observation.BaseObservation, reward: float, done: bool) \
            -> grid2op.Action.BaseAction:
        action, _ = self.act_with_id(observation)
        return action

    def reset(self, obs: grid2op.Observation.BaseObservation):
        self.next_actions = None

    def __get_actions(self, obs: grid2op.Observation.BaseObservation):
        # LOCAL: numpy evaluation of the Keras functional model's policy head, float32 as in TF.
        h = np.asarray(obs.to_vect(), dtype=np.float32)[: self.n_inputs].reshape((1, -1))
        for kernel, bias in self.layers:
            h = np.maximum(h @ kernel + bias, 0)
        logits = h @ self.head[0] + self.head[1]
        exp = np.exp(logits - logits.max(axis=1, keepdims=True))
        action_probability = (exp / exp.sum(axis=1, keepdims=True)).reshape(-1)
        sorted_actions = action_probability.argsort()[::-1]
        return sorted_actions.reshape(-1)
