"""Native Senior control/reward folding and terminal convention; deployment init/reset elsewhere."""
from typing import *
import numpy as np
from curriculumagent.common.utilities import find_best_line_to_reconnect,split_and_execute_action,revert_topo,set_bus_from_topo_vect
from curriculumagent.common.obs_converter import obs_to_vect

class NativeSeniorMethods:

    def run_next_action(self) -> np.ndarray:
        """
        Runs the environment until an action by the agent is required.

        In this function, the Grid2Op environment runs without agent intervention until the environment requires
        an action. The cumulative reward is calculated during this process, and the observation is returned when
        the agent is needed to take action.

        Returns:
            np.ndarray: The observation vector after the environment requires an action from the agent.
        """
        self.passiv_reward = 0
        continue_running = True
        cur_obs = self.single_env.get_obs()
        cumulated_rew = 0
        while continue_running:
            if cur_obs.rho.max() < self.action_threshold:
                if self.topo:
                    action_array = revert_topo(self.single_env.action_space, cur_obs)
                    default_action = self.single_env.action_space.from_vect(action_array)
                else:
                    default_action = self.single_env.action_space({})
                action = find_best_line_to_reconnect(obs=cur_obs, original_action=default_action)
                cur_obs, reward, _, self.info = self.single_env.step(action)
                cumulated_rew += reward
                self.step_in_env += 1
                if self.single_env.done:
                    continue_running = False
            else:
                continue_running = False
        self.obs_grid2op = cur_obs
        self.passiv_reward = cumulated_rew
        if isinstance(self.subset, list):
            obs_rl = self.obs_grid2op.to_vect()[self.subset]
        elif self.subset:
            obs_rl = obs_to_vect(self.obs_grid2op, False)
        else:
            obs_rl = self.obs_grid2op.to_vect()
        if self.scaler:
            if isinstance(self.scaler, BaseEstimator):
                obs_rl = self.scaler.transform(obs_rl.reshape(1, -1)).reshape(-1)
            elif isinstance(self.scaler, ObjectRef):
                obs_rl = ray.get(self.scaler).transform(obs_rl.reshape(1, -1)).reshape(-1)
        return obs_rl

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, Union[float, int], bool, bool, dict]:
        """
        Executes the agent's action in the environment and computes the reward.

        This method performs the action specified by the agent and accumulates rewards for passive actions taken
        by the environment. It checks if multiple steps are required and handles the transition based on the
        Grid2Op environment's conditions. The action can be executed sequentially if necessary.

        Args:
            action (np.ndarray): The action to execute, provided by the agent.

        Returns:
            Tuple[np.ndarray, Union[float, int], bool, bool, dict]: A tuple containing the following:
                - Observation vector as np.ndarray.
                - The accumulated reward as either float or int.
                - A boolean indicating if the episode is terminated.
                - A boolean indicating if the episode is truncated.
                - A dictionary containing additional environment info.
        """
        if self.single_env.done:
            logging.info('The Grid2Op Environment seems to be not initialized. We just return the lastobservation and do not run anything')
            self.done = True
            self.truncated, self.terminated = self.check_terminated_truncated()
            return (self.obs_rl, 0.0, self.terminated, self.truncated, self.info)
        if self.topology_actions:
            topo_vect = self.single_env.get_obs().topo_vect
            new_topology = self.actions[action]
            transf_act = set_bus_from_topo_vect(topo_vect, new_topology, self.single_env.action_space).to_vect()
        else:
            transf_act = self.actions[action]
        cur_obs, reward, self.done, self.info = split_and_execute_action(env=self.single_env, action_vect=transf_act)
        self.step_in_env += 1
        self.obs_grid2op = cur_obs
        if isinstance(self.subset, list):
            obs_rl = self.obs_grid2op.to_vect()[self.subset]
        elif self.subset:
            obs_rl = obs_to_vect(self.obs_grid2op, False)
        else:
            obs_rl = self.obs_grid2op.to_vect()
        if self.scaler:
            if isinstance(self.scaler, BaseEstimator):
                obs_rl = self.scaler.transform(obs_rl.reshape(1, -1)).reshape(-1)
            elif isinstance(self.scaler, ObjectRef):
                obs_rl = ray.get(self.scaler).transform(obs_rl.reshape(1, -1)).reshape(-1)
        self.obs_rl = obs_rl
        if cur_obs.rho.max() < self.action_threshold and self.done is False:
            self.obs_rl = self.run_next_action()
            reward += self.passiv_reward
        self.done = self.single_env.done
        self.truncated, self.terminated = self.check_terminated_truncated()
        game_over = 0
        if self.truncated:
            game_over = -300
        if self.terminated:
            game_over = 500
        self.reward = reward
        return (self.obs_rl, self.reward, self.terminated, self.truncated, self.info)

    def check_terminated_truncated(self) -> Tuple[bool, bool]:
        """
        Checks whether the environment is terminated or truncated.

        This method checks if the current episode is terminated or truncated based on the environment's status.
        Termination happens when the maximum episode duration is reached, while truncation occurs when the environment
        is finished early.

        Returns:
            Tuple[bool, bool]: A tuple containing two booleans:
                - The first boolean indicates if the episode is truncated.
                - The second boolean indicates if the episode is terminated.
        """
        if self.done:
            trunc, term = (True, False)
            if self.single_env.nb_time_step == self.single_env.chronics_handler.max_episode_duration():
                trunc, term = (False, True)
        else:
            trunc, term = (False, False)
        return (trunc, term)
