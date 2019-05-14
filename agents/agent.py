"""
Base Agent class, other agent classes inherited from this.
"""

import matplotlib.pyplot as plt
import numpy as np

from gym_tiadas.gym_tiadas.envs import Environment


class Agent:
    # Different icons
    __icons = {
        'BLANK': ' ', 'BLOCK': '■', 'FINAL': '$', 'CURRENT': '☺', 'UP': '↑', 'RIGHT': '→', 'DOWN': '↓', 'LEFT': '←',
        'STAY': '×'
    }

    def __init__(self, environment: Environment, epsilon: float = 0.1, gamma: float = 1., seed: int = 0,
                 states_to_observe: list = None, max_iterations: int = None):

        """
        :param environment: An environment where agent does any operation.
        :param epsilon: Epsilon using in e-greedy policy, to explore more states.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param states_to_observe: List of states from that we want to get a graphical output.
        :param max_iterations: Limits of iterations per episode.
        """

        # Discount factor
        assert 0 < gamma <= 1
        # Exploration factor
        assert 0 < epsilon <= 1

        self.gamma = gamma
        self.epsilon = epsilon

        # Set environment
        self.environment = environment

        # To intensive problems
        self.max_iterations = max_iterations
        self.iterations = 0

        # Create dictionary of states to observe
        if states_to_observe is None:
            self.states_to_observe = dict()
        else:
            self.states_to_observe = {state: list() for state in states_to_observe}

        # Current Agent State if the initial state of environment
        self.state = self.environment.initial_state

        # Initialize Random Generator with `seed` as initial seed.
        self.seed = seed
        self.generator = np.random.RandomState(seed=seed)

    def select_action(self, state: object = None) -> int:
        """
        Select best action with a little e-greedy policy.
        :return:
        """

        # If state is None, then set current state to state.
        if not state:
            state = self.state

        if self.generator.uniform(low=0., high=1.) < self.epsilon:
            # Get random action to explore possibilities
            action = self.environment.action_space.sample()

        else:
            # Get best action to exploit reward.
            action = self.best_action(state=state)

        return action

    def episode(self) -> None:
        """
        Run an episode complete until get a final step
        :return:
        """
        raise NotImplemented

    def reset(self) -> None:
        """
        Reset agent, forgetting previous q-values
        :return:
        """
        raise NotImplemented

    def best_action(self, state: object = None) -> int:
        """
        Return best action a state given.
        :return:
        """
        raise NotImplemented

    def reset_iterations(self) -> None:
        """
        Set iterations to zero.
        :return:
        """
        self.iterations = 0

    def show_observed_states(self) -> None:
        """
        Show graph of observed states
        :return:
        """
        for state, data in self.states_to_observe.items():
            plt.plot(data, label='State: {}'.format(state))

        plt.xlabel('Iterations')
        plt.ylabel('V max')

        plt.legend(loc='upper left')

        plt.show()

    def print_information(self) -> None:
        """
        Print basic information about agent
        :return:
        """
        print('- Agent information! -')
        print("Seed: {}".format(self.seed))
        print("Gamma: {}".format(self.gamma))
        print("Epsilon: {}".format(self.epsilon))

    def train(self, epochs=1000):
        """
        Return this agent trained with `epochs` epochs.
        :param epochs:
        :return:
        """

        for _ in range(epochs):
            # Do an episode
            self.episode()
