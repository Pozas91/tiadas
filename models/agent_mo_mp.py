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

import utils.hypervolume as uh
import utils.miscellaneous as um
from .vector import Vector
from .vector_float import VectorFloat


class AgentMOMP:
    # JSON indent
    json_indent = 2
    # Get dumps path from this file path
    dumps_path = '{}/../dumps'.format(os.path.dirname(os.path.abspath(__file__)))

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

        # Current agent state is initial_state in the environment
        self.state = self.environment.initial_state

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

        # Evaluation mechanism
        if evaluation_mechanism in ('HV-PQL', 'PO-PQL', 'C-PQL'):
            self.evaluation_mechanism = evaluation_mechanism
        else:
            raise ValueError('Evaluation mechanism does not valid.')

        # Default reward agent
        self.default_reward = self.environment.default_reward * 0

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
        r_s_a = r_s_a_dict.get(action, self.default_reward + 0)
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
        r_s_a = self.r.get(state, {}).get(action, self.default_reward)

        # Get ND(s, a)
        non_dominated_vectors = self.nd.get(state, {}).get(action, [self.default_reward + 0])
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

    def track_policy(self, state, target):
        """
        This method track a policy until reach target given.
        :param state: initial state
        :param target: a policy to follow
        :return:
        """

        path = [state]
        last_state = state

        # While s is not terminal
        while not self.environment.is_final(state=state):

            # Path found
            found = False

            # For each a in environments actions.
            for a in self.environment.actions.values():

                # If path is found
                if found:
                    break

                # Retrieve R(s, a)
                r = self.r.get(state).get(a)

                # Retrieve ND(s, a)
                nd = self.nd.get(state).get(a)

                for q in nd:

                    # Calc vector to follow
                    v = (q * self.gamma) + r

                    # if are equals with relaxed equality operator
                    if v.all_close(target):
                        # This transaction must be determinist s': T(s'|s, a) = 1
                        state = self.environment.next_state(action=a, state=state)
                        target = q

                        # Append to path
                        path.append(state)

                        # Path found
                        found = True

                        break

            if state == last_state:
                raise ValueError('Path not found from state: {} and target: {}.'.format(state, target))

        return path

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
            hv.append(uh.calc_hypervolume(list_of_vectors=q_set, reference=self.hv_reference))

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
            evaluation = uh.calc_hypervolume(list_of_vectors=q_set, reference=self.hv_reference)

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
            non_dominated = self.default_reward.m3_max(q_set)

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
            non_dominated = self.default_reward.m3_max(q_set)

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
        return json.dumps(model, indent=self.json_indent)

    def save(self, filename=None):
        """
        Save model into json file.
        :param filename: If is None, then get current timestamp as filename (defaults 'dumps' dir).
        :return:
        """

        if filename is None:
            # Get environment name in snake case
            environment = um.str_to_snake_case(self.environment.__class__.__name__)

            # Get evaluation mechanism in snake case
            evaluation_mechanism = um.str_to_snake_case(self.evaluation_mechanism)

            # Prepare file name
            filename = '{}_{}_{}'.format(environment, evaluation_mechanism, datetime.datetime.now().timestamp())

        # Prepare file path
        file_path = AgentMOMP.dumps_file_path(filename=filename)

        # Get dict model
        model = self.get_dict_model()

        # Open file with filename in write mode with UTF-8 encoding.
        with open(file_path, 'w', encoding='UTF-8') as file:
            json.dump(model, file, indent=self.json_indent)

    def non_dominate_vectors_from_state(self, state):
        """
        Return all non dominate vectors from state given.
        :param state:
        :return:
        """

        # Get list of non dominate vectors
        nd_lists = self.nd.get(state, {}).values()

        # Flatten nd_lists
        nd = [item for sublist in nd_lists for item in sublist]

        return self.default_reward.m3_max(nd)

    @staticmethod
    def load(filename=None, environment=None, evaluation_mechanism=None):
        """
        Load json string from file and convert to dictionary.
        :param evaluation_mechanism: It is an evaluation mechanism that you want load
        :param environment: It is an environment that you want load.
        :param filename: If is None, then get last timestamp file from 'dumps' dir.
        :return:
        """

        # Check if filename is None
        if filename is None:

            # Check if environment is also None
            if environment is None:
                raise ValueError('If you has not indicated a filename, you must indicate a environment.')

            # Check if evaluation mechanism is also None
            if evaluation_mechanism is None:
                raise ValueError('If you has not indicated a filename, you must indicate a evaluation mechanism.')

            # Get environment name in snake case
            environment = um.str_to_snake_case(environment.__name__)

            # Get evaluation mechanism name in snake case
            evaluation_mechanism = um.str_to_snake_case(evaluation_mechanism)

            # Filter str
            filter_str = '{}_{}'.format(environment, evaluation_mechanism)

            for root, directories, files in os.walk(AgentMOMP.dumps_path):

                # Filter files with that environment and evaluation mechanism
                files = filter(lambda f: filter_str in f, files)

                # Files to list
                files = list(files)

                # Sort list of files
                files.sort()

                # At least must have a file
                if files:
                    # Get last filename
                    filename = files[-1].split('.json')[0]

        # Prepare file path
        file_path = AgentMOMP.dumps_file_path(filename)

        # Read file from path
        try:
            file = open(file_path, 'r', encoding='UTF-8')
        except FileNotFoundError:
            return None

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

    @staticmethod
    def dumps_file_path(filename):
        # Return path from file name
        return '{}/{}.json'.format(AgentMOMP.dumps_path, filename)
