"""
Agent multi-objective multi-policy.
Save rewards, N and occurrences independently, to calculate Q-set on runtime.

Evaluation mechanisms available
    - HV-PQL: Based in hypervolume
    - C-PQL: Based in cardinality
    - PO-PQL: Based in Pareto

For take instances of default_reward, we multiply by zero default_reward, that take less time that other operations such
as deepcopy (≈ 247% faster) or copy (≈ 118% faster).
"""
import datetime
import importlib
import json
import os

import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import utils.miscellaneous as um
import utils.pareto as up
from .vector import Vector
from .vector_float import VectorFloat


class AgentMOMP:
    # JSON indent
    JSON_INDENT = 2

    def __init__(self, environment, epsilon=0.1, gamma=1., seed=0, max_iterations=None,
                 hv_reference=None, evaluation_mechanism='HV-PQL', states_to_observe=None):

        # Discount factor
        assert 0 < gamma <= 1
        # Exploration factor
        assert 0 < epsilon <= 1

        self.epsilon = epsilon
        self.gamma = gamma

        # Set environment
        self.environment = environment

        # To intensive problems
        self.max_iterations = max_iterations
        self.iterations = 0

        # Create dictionary of states to observe
        if states_to_observe is None:
            self.states_to_observe = dict()
        else:
            self.states_to_observe = {
                state: list() for state in states_to_observe
            }

        # Current Agent State if the initial state of environment
        self.state = self.environment.reset()

        # Initialize Random Generator with `seed` as initial seed.
        self.seed = seed
        self.generator = np.random.RandomState(seed=self.seed)

        # Average observed immediate reward vector.
        self.r = dict()

        # Set of non-dominated vectors
        self.nd = dict()

        # Occurrences
        self.n = dict()

        # HV reference
        self.hv_reference = hv_reference

        # Check if evaluation mechanism selected is available
        if evaluation_mechanism in ('HV-PQL', 'C-PQL', 'PO-PQL'):
            self.evaluation_mechanism = evaluation_mechanism
        else:
            raise ValueError('{} evaluation mechanism not recognize!'.format(evaluation_mechanism))

        # Default reward
        self.default_reward = self.environment.default_reward

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
        r_s_a = r_s_a_dict.get(action, self.environment.default_reward)
        # Update R(s, a)
        r_s_a += (reward - r_s_a) / occurrences
        # Update with new R(s, a)
        r_s_a_dict.update({action: r_s_a})
        # Update R dictionary
        self.r.update({state: r_s_a_dict})

    def update_nd_s_a(self, state, action, next_state):

        # Union
        union = list()

        # for each action in actions
        for a in self.environment.actions.values():
            # Get Q(s, a)
            q = self.q_set(state=next_state, action=a)

            # Union with flatten mode.
            union += q

        # Update ND policies of s' in s
        nd_s_a = self.environment.default_reward.m3_max(union)

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
        r_s_a = self.r.get(state, {}).get(action, self.environment.default_reward)

        # Get ND(s, a)
        non_dominated_vectors = self.nd.get(state, {}).get(action, [self.environment.default_reward])
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
        Reset agent, forgetting previous dictionaries
        :return:
        """
        self.r = dict()
        self.nd = dict()
        self.n = dict()
        self.state = self.environment.reset()
        self.iterations = 0

        # Reset states to observe
        for state in self.states_to_observe:
            self.states_to_observe.update({
                state: list()
            })

    def reset_iterations(self):
        """
        Set iterations to zero.
        :return:
        """
        self.iterations = 0

    def best_action(self, state=None):
        """
        Return best action for q and state given.
        :param state:
        :return:
        """

        # if don't specify a state, get current state.
        if not state:
            state = self.state

        # Use the selected evaluation
        if self.evaluation_mechanism == 'HV-PQL':
            action = self.hypervolume_evaluation(state=state)
        elif self.evaluation_mechanism == 'C-PQL':
            action = self.cardinality_evaluation(state=state)
        else:
            action = self.pareto_evaluation(state=state)

        return action

    @staticmethod
    def track_policy(state, target):
        pass

    def _best_hypervolume(self, state):
        """
        Return best hypervolume for state given.
        :param state:
        :return:
        """

        hv = list()

        for a in self.environment.actions.values():
            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=state, action=a)

            # Calc hypervolume of Q_set, with reference given.
            hv.append(up.hypervolume(vector=q_set, reference=self.hv_reference))

        return max(hv)

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

    def hypervolume_evaluation(self, state):
        """
        Calc the hypervolume for each action in state given. (HV-PQL)
        :param state:
        :return:
        """

        actions = list()
        max_evaluation = 0

        # for each a in actions
        for a in self.environment.actions.values():

            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=state, action=a)

            # Calc hypervolume of Q_set, with reference given.
            evaluation = up.hypervolume(vector=q_set, reference=self.hv_reference)

            # If current value is close to new value
            if math.isclose(a=evaluation, b=max_evaluation):
                # Append another possible action
                actions.append(a)

            elif evaluation > max_evaluation:
                # Create a new list with current key.
                actions = [a]

            # Update max value
            max_evaluation = max(max_evaluation, evaluation)

        # from best actions get one aleatory.
        return self.generator.choice(actions)

    def cardinality_evaluation(self, state):
        """
        Calc the cardinality for each action in state given. (C-PQL)
        :param state:
        :return:
        """

        actions = list()
        max_evaluation = 0

        # for each a in actions
        for a in self.environment.actions.values():

            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=state, action=a)

            # Use m3_max algorithm to get non_dominated vectors from q_set.
            non_dominated = self.environment.default_reward.m3_max(q_set)

            # Get number of non_dominated vectors.
            evaluation = len(non_dominated)

            # If current value is equal to new value
            if evaluation == max_evaluation:
                # Append another possible action
                actions.append(a)

            elif evaluation > max_evaluation:
                # Create a new list with current key.
                actions = [a]

            # Update max value
            max_evaluation = max(max_evaluation, evaluation)

        # from best actions get one aleatory.
        return self.generator.choice(actions)

    def pareto_evaluation(self, state):
        """
        Calc the pareto for each action in state given. (PO-PQL)
        :param state:
        :return:
        """

        actions = list()

        # for each a in actions
        for a in self.environment.actions.values():
            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=state, action=a)
            # Use m3_max algorithm to get non_dominated vectors from q_set.
            non_dominated = self.environment.default_reward.m3_max(q_set)

            # If has almost one non_dominated vector is a valid action.
            if len(non_dominated) > 0:
                actions.append(a)

        # If actions is empty, get all possible actions.
        if not actions:
            actions = self.environment.actions.values()

        # Get random action from valid action.
        return self.generator.choice(actions)

    def get_dict_model(self):
        """
        Get a dictionary of model
        In JSON serialize only is valid strings as key on dict, so we convert all numeric keys in strings keys.
        :return:
        """
        model = {
            'meta': {
                'class': self.__class__.__name__,
                'module': self.__module__,
                'dependencies': {
                    'numpy': np.__version__,
                    'matplotlib': matplotlib.__version__
                }
            },
            'data': {
                'epsilon': self.epsilon,
                'gamma': self.gamma,
                'environment': {
                    'meta': {
                        'class': self.environment.__class__.__name__,
                        'module': self.environment.__module__,
                    },
                    'data': self.environment.get_dict_model()
                },
                'max_iterations': self.max_iterations,
                'states_to_observe': [{'key': k, 'value': v} for k, v in self.states_to_observe.items()],
                'seed': self.seed,
                'r': [
                    {'key': k, 'value': {'key': int(k2), 'value': v2.tolist()}}
                    for k, v in self.r.items() for k2, v2 in v.items()
                ],
                'nd': [
                    {'key': k, 'value': {'key': int(k2), 'value': [vector.tolist() for vector in v2]}}
                    for k, v in self.nd.items() for k2, v2 in v.items()
                ],
                'n': [
                    {'key': k, 'value': {'key': int(k2), 'value': v2}}
                    for k, v in self.n.items() for k2, v2 in v.items()
                ],
                'hv_reference': self.hv_reference.tolist(),
                'evaluation_mechanism': self.evaluation_mechanism
            }
        }

        return model

    def to_json(self):
        """
        Get a dict model from current object and return as json string.
        :return:
        """
        model = self.get_dict_model()
        return json.dumps(model, indent=self.JSON_INDENT)

    def save(self, filename=None):
        """
        Save model into json file.
        :param filename: If is None, then get current timestamp as filename (defaults 'dumps' dir).
        :return:
        """

        if filename is None:
            filename = datetime.datetime.now().timestamp()

        file_path = '../dumps/{}.json'.format(filename)

        model = self.get_dict_model()

        # Open file with filename in write mode with UTF-8 encoding.
        with open(file_path, 'w', encoding='UTF-8') as file:
            json.dump(model, file, indent=self.JSON_INDENT)

    @staticmethod
    def load(filename=None):
        """
        Load json string from file and convert to dictionary.
        :param filename: If is None, then get last timestamp file from 'dumps' dir.
        :return:
        """

        if filename is None:
            path = '../dumps'

            for root, directories, files in os.walk(path):
                # Sort list of files
                files.sort()

                # Get last filename
                filename = files[-1].split('.json')[0]

        file_path = '../dumps/{}.json'.format(filename)

        with open(file_path, 'r', encoding='UTF-8') as file:
            # Load structured data from indicated file.
            model = json.load(file)

        # Get meta-data
        meta = model.get('meta')
        # Get data
        data = model.get('data')

        # ENVIRONMENT
        environment = data.get('environment')

        # Meta
        environment_meta = environment.get('meta')
        environment_class_name = environment_meta.get('class')
        environment_module_name = environment_meta.get('module')
        environment_module = importlib.import_module(environment_module_name)
        environment_class_ = getattr(environment_module, environment_class_name)

        # Data
        environment_data = environment.get('data')

        # Instance
        environment = environment_class_()

        # Set environment data
        for key, value in environment_data.items():

            if 'state' in key or 'transactions' in key:
                value = tuple(value)

            elif 'default_reward' in key:
                # If all elements are int, then default_reward is a integer Vector, otherwise float Vector
                value = Vector(value) if (all([isinstance(x, int) for x in value])) else VectorFloat(value)

            vars(environment)[key] = value

        # Set seed
        environment.seed(seed=environment.initial_seed)

        # Get default reward as reference
        default_reward = environment.default_reward

        # AgentMOMP

        # Meta

        # Prepare module and class to make an instance.
        class_name = meta.get('class')
        module_name = meta.get('module')
        module = importlib.import_module(module_name)
        class_ = getattr(module, class_name)

        # Data
        epsilon = data.get('epsilon')
        gamma = data.get('gamma')
        max_iterations = data.get('max_iterations')
        seed = data.get('seed')
        evaluation_mechanism = data.get('evaluation_mechanism')

        # default_reward is reference so, reset components (multiply by zero) and add hv_reference to get hv_reference.
        hv_reference = (default_reward * 0) + data.get('hv_reference')

        # Update 'states_to_observe' data
        states_to_observe = dict()
        for item in data.get('states_to_observe'):
            key = tuple(item.get('key'))
            value = item.get('value')

            states_to_observe.update({key: value})

        # Unpack 'r' data
        r = dict()
        for item in data.get('r'):

            key = um.lists_to_tuples(item.get('key'))
            value = item.get('value')
            action = value.get('key')
            value = (default_reward * 0) + value.get('value')

            if key not in r.keys():
                r.update({key: dict()})

            r.get(key).update({action: value})

        # Unpack 'nd' data
        nd = dict()
        for item in data.get('nd'):

            key = um.lists_to_tuples(item.get('key'))
            value = item.get('value')
            action = value.get('key')
            value = [(default_reward * 0) + v for v in value.get('value')]

            if key not in nd.keys():
                nd.update({key: dict()})

            nd.get(key).update({action: value})

        # Unpack 'n' data
        n = dict()
        for item in data.get('n'):

            key = um.lists_to_tuples(item.get('key'))
            value = item.get('value')
            action = value.get('key')
            value = int(value.get('value'))

            if key not in n.keys():
                n.update({key: dict()})

            n.get(key).update({action: value})

        # Prepare an instance of model.
        model = class_(environment=environment, epsilon=epsilon, gamma=gamma, seed=seed,
                       max_iterations=max_iterations, hv_reference=hv_reference,
                       evaluation_mechanism=evaluation_mechanism)

        # Set finals settings and return it.
        model.r = r
        model.nd = nd
        model.n = n
        model.states_to_observe = states_to_observe

        return model
