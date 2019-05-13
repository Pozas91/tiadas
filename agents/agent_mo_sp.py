"""
Agent multi-objective single-policy.
Convert the rewards vector into a scalarized reward, after that use Q-Learning method. It follow same process that agent
model, but the reward its calculate multiply the weights vector and the rewards vector.
"""
import numpy as np

from gym_tiadas.gym_tiadas.envs import Environment
from models import Vector
from .agent import Agent


class AgentMOSP(Agent):

    def __init__(self, environment: Environment, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 1.,
                 seed: int = 0, states_to_observe: list = None, max_iterations: int = None, weights: tuple = None):
        """
        :param environment: An environment where agent does any operation.
        :param alpha: Learning rate
        :param epsilon: Epsilon using in e-greedy policy, to explore more states.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param states_to_observe: List of states from that we want to get a graphical output.
        :param max_iterations: Limits of iterations per episode.
        :param weights: Tuple of weights to multiply per reward vector.
        """

        # Sum of weights must be 1.
        assert weights is not None and np.sum(weights) == 1.

        # Super call init
        super().__init__(environment=environment, alpha=alpha, epsilon=epsilon, gamma=gamma, seed=seed,
                         states_to_observe=states_to_observe, max_iterations=max_iterations)

        # Set weights
        self.weights = weights

    def process_reward(self, reward: Vector) -> Vector:
        """
        Processing reward function.
        :param reward:
        :return:
        """

        # Multiply the reward for the vector weights, sum all components and return a reward of the same type as the
        # original, but with only one component.
        return reward.__class__(float(np.sum(reward * self.weights)))

    def _update_q_dictionary(self, reward: Vector, action: int, next_state: object) -> None:
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
