"""
Agent multi-objective multi-policy.
Save rewards, N and occurrences independently, to calculate Q-set on runtime.
"""

from copy import deepcopy

import math
import matplotlib.pyplot as plt
import numpy as np

import utils.pareto as up


class AgentMOMP:

    def __init__(self, environment, default_reward, epsilon=0.1, gamma=1., seed=0, states_to_observe=None,
                 max_iterations=None):

        # Discount factor
        assert 0 < gamma <= 1
        # Exploration factor
        assert 0 < epsilon <= 1

        self.epsilon = epsilon
        self.gamma = gamma

        # Default reward
        self.default_reward = default_reward
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
        self.state = self.environment.reset()

        # Initialize Random Generator with `seed` as initial seed.
        self.generator = np.random.RandomState(seed=seed)

        # Average observed immediate reward vector.
        self.r = dict()

        # Set of non-dominated vectors
        self.nd = dict()

        # Occurrences
        self.n = dict()

    def episode(self) -> None:
        """
        Run an episode complete until get a final step.
        :return:
        """

        # Reset environment
        self.state = self.environment.reset()

        # Condition to stop episode
        is_final = False

        # Reset iterations
        self.reset_iterations()

        while not is_final:
            # Increment iterations
            self.iterations += 1

            # Choose action a from s using a policy derived from the Q-set
            action = self.select_action()

            # Do step on environment
            next_state, rewards, is_final, info = self.environment.step(action=action)

            # Convert reward to an vector (This operation is for do not declare a type of vector)
            rewards = (self.default_reward * 0) + rewards

            # Update ND policies of s' in s
            self.update_nd_s_a(state=self.state, action=action, next_state=next_state)

            # Update numbers of occurrences
            n_s_a = self.get_and_update_n_s_a(state=self.state, action=action)

            # Update average immediate rewards
            self.update_r_s_a(state=self.state, action=action, reward=rewards, occurrences=n_s_a)

            # Proceed to next state
            self.state = next_state

        # Append new data
        for state, data in self.states_to_observe.items():
            # Add to data Best value (V max)
            value = self._best_hypervolume(state)

            # Add to data Best value (V max)
            data.append(value)

            # Update dictionary
            self.states_to_observe.update({state: data})

    def get_and_update_n_s_a(self, state, action) -> int:
        """
        Update n(s, a) dictionary.
        :param state:
        :param action:
        :return:
        """

        # Get n(s) dict
        n_s_a_dict = self.n.get(state, {})
        # Get n(s, a) value
        n_s_a = (n_s_a_dict.get(action, 0) + 1)
        # Update with the increment.
        n_s_a_dict.update({action: n_s_a})
        # Update dictionary
        self.n.update({state: n_s_a_dict})

        return n_s_a

    def update_r_s_a(self, state, action, reward, occurrences):

        # Get R(s) dict.
        r_s_a_dict = self.r.get(state, {})
        # Get R(s, a) value.
        r_s_a = r_s_a_dict.get(action, deepcopy(self.default_reward))
        # Update R(s, a)
        r_s_a += (reward - r_s_a) / occurrences
        # Update R dictionary
        r_s_a_dict.update({action: r_s_a})
        self.r.update({state: r_s_a_dict})

    def update_nd_s_a(self, state, action, next_state):

        # Union
        union = list()

        # for each action in actions
        for a in self.environment.actions.values():
            q = self.q_set(state=next_state, action=a)

            # Union with flatten mode.
            union += q

        # Update ND policies of s' in s (
        nd_s_a = self.default_reward.m3_max(union)

        # ND(s, a) <- (ND(U_a' Q_set(s', a'))
        nd_s_a_dict = self.nd.get(state, {})
        nd_s_a_dict.update({action: nd_s_a})
        self.nd.update({state: nd_s_a_dict})

    def q_set(self, state, action):
        """
        Calc on run-time Q-set
        :param state:
        :param action:
        :return:
        """

        # Get R(s, a) with default.
        r_s_a = self.r.get(state, {}).get(action, deepcopy(self.default_reward))

        # Get ND(s, a)
        non_dominated_vectors = self.nd.get(state, {}).get(action, [deepcopy(self.default_reward)])
        q_set = list()

        # R(s, a) + y*ND
        for non_dominated in non_dominated_vectors:
            q_set.append(r_s_a + (non_dominated * self.gamma))

        return q_set

    def select_action(self, state=None) -> int:
        """
        Select best action with an e-greedy policy.
        :return:
        """

        # If state is None, then set current state to state.
        if not state:
            state = self.state

        if self.generator.uniform(low=0, high=1) < self.epsilon:
            # Get random action to explore possibilities
            action = self.environment.action_space.sample()
        else:
            # Get best action to exploit reward.
            action = self.best_action(state=state)

        return action

    def reset(self):
        """
        Reset agent, forgetting previous q-values
        :return:
        """
        self.r = dict()
        self.nd = dict()
        self.n = dict()
        self.state = self.environment.reset()
        self.iterations = 0

    def reset_iterations(self):
        """
        Set iterations to zero.
        :return:
        """
        self.iterations = 0

    def hypervolume_evaluation(self, state):
        """
        Calc the hypervolume for each action in state given. (HV-PQL)
        :return:
        """

        evaluations = list()

        # Get all vectors to get a common reference
        vectors = {a: self.q_set(state=state, action=a) for a in self.environment.actions.values()}

        # Calc reference point (Get min axis of all vectors and subtract 1 for reference point)
        reference = (np.min([vector for vector_list in vectors.values() for vector in vector_list], axis=0) - 1)

        # for each vector in vectors dictionary
        for a, vector in vectors.items():
            hv_a = up.hypervolume(vector=vector, reference=reference)
            evaluations.insert(a, hv_a)

        return evaluations

    def cardinality_evaluation(self, state):
        """
        Calc the cardinality for each action in state given. (C-PQL)
        :param state:
        :return:
        """

        evaluations = list()

        for a in self.environment.actions.values():
            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=state, action=a)
            # Use m3_max algorithm to get non_dominated vectors from q_set.
            non_dominated = self.default_reward.m3_max(q_set)
            # Get number of non_dominated vectors.
            evaluations.insert(a, len(non_dominated))

        return evaluations

    def best_action(self, state=None):
        """
        Return best action for q and state given.
        :param state:
        :return:
        """

        # if don't specify a state, get current state.
        if not state:
            state = self.state

        # Use hypervolume evaluation
        # evaluations = self.hypervolume_evaluation(state=state)
        evaluations = self.cardinality_evaluation(state=state)

        # Initialize actions with last action
        actions = [len(evaluations) - 1]

        # Get last element of list
        max_evaluation = evaluations.pop()

        # for each evaluation
        for action, evaluation in enumerate(evaluations):

            # If current value is close to new value
            if math.isclose(a=evaluation, b=max_evaluation):
                # Append another possible action
                actions.append(action)

            elif evaluation > max_evaluation:
                # Create a new list with current key.
                actions = [action]

            # Update max value
            max_evaluation = max(max_evaluation, evaluation)

        # from best actions get one aleatory.
        return self.generator.choice(actions)

    @staticmethod
    def track_policy(state, target):
        pass

    def _best_hypervolume(self, state):
        """
        Return best hypervolume for state given.
        :param state:
        :return:
        """
        return max(self.hypervolume_evaluation(state=state))

    def print_observed_states(self):
        """
        Show graph of observed states
        :return:
        """
        for state, data in self.states_to_observe.items():
            plt.plot(data, label='State: {}'.format(state))

        plt.xlabel('Iterations')
        plt.ylabel('HV max')

        plt.legend(loc='upper left')

        plt.show()
