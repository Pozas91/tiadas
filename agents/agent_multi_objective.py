"""
It follow same process that agent model, but the reward its calculate multiply the vector of weights and the
vector of rewards.
"""

import numpy as np

from agents import Agent


class AgentMultiObjective(Agent):

    def __init__(self, environment, alpha=0.1, epsilon=0.1, gamma=1., seed=0, default_reward=0.,
                 states_to_observe=None, max_iterations=None, weights=None):
        # Sum of weights must be 1.
        assert weights is not None and np.sum(weights) == 1.

        # Super call init
        super().__init__(environment, alpha, epsilon, gamma, seed, default_reward, states_to_observe,
                         max_iterations)

        # Set weights
        self.weights = weights

    def process_reward(self, reward) -> float:
        """
        Processing reward function.
        :param reward:
        :return:
        """

        # Apply weights to rewards to get only one reward
        return float(np.sum(np.multiply(reward, self.weights)))

    def _update_q_dictionary(self, reward, action, next_state) -> None:
        """
        Update Q-Dictionary with new data
        :param reward:
        :param action:
        :param next_state:
        :return:
        """

        # Apply function
        reward = self.process_reward(reward=reward)

        # Super call
        super()._update_q_dictionary(reward=reward, action=action, next_state=next_state)
