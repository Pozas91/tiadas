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

    def pareto_q_learning(self, epochs=1000):
        """

        :param epochs:
        :return:
        """

        # Reset mesh
        self.reset()

        # Rewards
        r = dict()

        # For each epoch
        for t in range(epochs):
            # Initialize state s
            self.state = self.environment.reset()

            # Condition to stop an episode
            is_final_state = False

            # Reset iterations
            self.reset_iterations()

            # Until s is terminal
            while not is_final_state:
                # Increment iterations
                self.iterations += 1

                # Get an action
                action = self.select_action()

                # Do step on environment
                next_state, reward, is_final_state, info = self.environment.step(action=action)

                # Update ND policies of s' in s

                # Update average immediate rewards
                old_r = self.q.get(self.state, {}).get(action, self.default_reward)

                update_r = {action: np.sum()}
                r.get(self.state).update({
                    action: r.get(self.state, {}).get(action, 0.0)
                })

                # Update Q-Dictionary
                self._update_q_dictionary(reward=reward, action=action, next_state=next_state)

                # Update state
                self.state = next_state

                # Check timeout
                if self.max_iterations is not None and not is_final_state:
                    is_final_state = self.iterations >= self.max_iterations
