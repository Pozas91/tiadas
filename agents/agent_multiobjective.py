"""
It follow same process that agent model, but the reward its calculate multiply the vector of weights and the
vector of rewards.
"""

import numpy as np

from agents import Agent


class AgentMultiObjective(Agent):

    def __init__(self, environment, alpha=0.1, epsilon=0.1, gamma=1., seed=0, default_reward=0.,
                 states_to_observe=None, max_iterations=None, weights=None, number_of_rewards=2):
        super().__init__(environment, alpha, epsilon, gamma, seed, default_reward, states_to_observe,
                         max_iterations)

        # If not weights define, all rewards have same weight
        self.weights = [1. / number_of_rewards] * number_of_rewards if weights is None else weights

        # Sum of weights must be 1.
        assert np.sum(self.weights) == 1.

    def set_rewards_weights(self, rewards_weights) -> None:
        """
        Set weights
        :param rewards_weights:
        :return:
        """
        self.weights = rewards_weights

    def _processing_reward(self, reward):
        """
        Processing reward function.
        :param reward:
        :return:
        """

        # Apply weights to rewards to get only one reward
        return np.sum(np.multiply(reward, self.weights))
