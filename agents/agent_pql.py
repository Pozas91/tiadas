"""
Agent PQL.

Implementation of the PQ-learning algorithm by Kristof Van Moffaert and Ann Nowé, in "Multi-Objective Reinforcement
Learning using Sets of Pareto Dominating Policies" paper.

Save rewards, N and occurrences independently, to calculate Q-set on runtime.

Evaluation mechanisms available
    - HV-PQL: Based in hypervolume
    - C-PQL: Based in cardinality
    - PO-PQL: Based in Pareto
    
    
Sample call: 
    
    1) train
    
        # Instance of environment
        env = DeepSeaTreasure()

        # Instance of agent
        agent = AgentPQL(environment=env)

        # Train agent
        agent.train() # Optional you can pass a number of epochs, e.g. agent.train(epochs=3000)
    
    
    2) write agent to file
    
        # Instance of environment
        env = DeepSeaTreasure()

        # Instance of agent
        agent = AgentPQL(environment=env)

        # Write to file
        agent.save() # Optional you can pass a filename, e.g. agent.save(filename='my_agent')
    
    
    3) read agent from file
    
        # Instance of environment
        env = DeepSeaTreasure()

        # Evaluation mechanism
        evaluation_mechanism = 'C-PQL'

        # Recover agent with that features
        agent = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Optional you can pass a filename
        agent = AgentPQL.load(filename='my_agent')
    

NOTE: Each environment defines its default reward, either integer or float.
The agent uses a default value of zero for various operations. In order to get
this zero vector of the same length and type of the default reward, we
multiply the default reward by zero using our defined operation * for vectors.
This seems faster than using either deepcopy (≈ 247% faster) or copy 
(≈ 118% faster).
"""
import datetime
import importlib
import json
import math
import os
from copy import deepcopy

import matplotlib
import numpy as np

import utils.hypervolume as uh
import utils.miscellaneous as um
from gym_tiadas.gym_tiadas.envs import Environment
from models import ActionVector
from models.vector import Vector
from models.vector_float import VectorFloat
from .agent import Agent


class AgentPQL(Agent):
    # Indent of the JSON file where the agent will be saved
    json_indent = 2
    # Get dumps path from this file path
    dumps_path = '{}/../dumps'.format(os.path.dirname(os.path.abspath(__file__)))

    def __init__(self, environment: Environment, epsilon: float = 0.1, gamma: float = 1., seed: int = 0,
                 max_iterations: int = None, hv_reference: Vector = None, evaluation_mechanism: str = 'HV-PQL',
                 states_to_observe: list = None):
        """
        :param environment: instance of any environment class.
        :param epsilon: Epsilon used in epsilon-greedy policy, to explore more states.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param max_iterations: Limits of iterations per episode.
        :param hv_reference: Reference vector to calc hypervolume
        :param evaluation_mechanism: Evaluation mechanism used to calc best action to choose. Three values are
            available: 'C-PQL', 'PO-PQL', 'HV-PQL'
        :param states_to_observe: List of states from that we want to get a graphical output.
        """

        # Super call __init__
        super().__init__(environment=environment, epsilon=epsilon, gamma=gamma, seed=seed,
                         states_to_observe=states_to_observe, max_iterations=max_iterations)

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

    def episode(self) -> None:
        """
        Run an complete episode using the agent's epsilon-greedy policy
        until a final state is reached.
        The agent updates R(s,a), ND(s,a) and n(s,a) values.
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

            # Check if is final
            if is_final:
                # ND(s, a) <- Zero vector
                nd_s_a_dict = self.nd.get(self.state, {})
                nd_s_a_dict.update({action: [self.environment.default_reward.zero_vector]})
                self.nd.update({self.state: nd_s_a_dict})
            else:
                # Update ND policies of s' in s
                self.update_nd_s_a(state=self.state, action=action, next_state=next_state)

            # Update numbers of occurrences
            n_s_a = self.get_and_update_n_s_a(state=self.state, action=action)

            # Update average immediate rewards
            self.update_r_s_a(state=self.state, action=action, reward=rewards, occurrences=n_s_a)

            # Proceed to next state
            self.state = next_state

            # Check timeout
            if self.max_iterations is not None and not is_final:
                is_final = self.iterations >= self.max_iterations

        # Append new data for graphical output
        for state, data in self.states_to_observe.items():

            # Set state
            self.environment.current_state = state

            # Add to data Best value (V max)
            value = self._best_hypervolume()

            # Add to data Best value (V max)
            data.append(value)

            # Update dictionary
            self.states_to_observe.update({state: data})

    def get_and_update_n_s_a(self, state: object, action: int) -> int:
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

    def update_r_s_a(self, state: object, action: int, reward: Vector, occurrences: int) -> None:
        """
        Update r(s, a) dictionary.
        :param state:
        :param action:
        :param reward:
        :param occurrences:
        :return:
        """

        # Get R(s) dict.
        r_s_a_dict = self.r.get(state, {})
        # Get R(s, a) value.
        r_s_a = r_s_a_dict.get(action, self.environment.default_reward.zero_vector)
        # Update R(s, a)
        r_s_a += (reward - r_s_a) / occurrences
        # Update with new R(s, a)
        r_s_a_dict.update({action: r_s_a})
        # Update R dictionary
        self.r.update({state: r_s_a_dict})

    def update_nd_s_a(self, state: object, action: int, next_state: object) -> None:
        """
        Update ND(s, a)
        :param state:
        :param action:
        :param next_state:
        :return:
        """

        # Union
        union = list()

        # for each action in actions
        for a in self.environment.action_space:
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

    def q_set(self, state: object, action: int) -> list:
        """
        Calc on run-time Q-set
        :param state:
        :param action:
        :return:
        """

        # Get R(s, a) with default.
        r_s_a = self.r.get(state, {}).get(action, self.environment.default_reward.zero_vector)

        # Get ND(s, a)
        non_dominated_vectors = self.nd.get(state, {}).get(action, [self.environment.default_reward.zero_vector])
        q_set = list()

        # R(s, a) + y*ND
        for non_dominated in non_dominated_vectors:
            q_set.append(r_s_a + (non_dominated * self.gamma))

        return q_set

    def reset(self) -> None:
        """
        Reset agent, forgetting previous dictionaries
        :return:
        """
        self.r = dict()
        self.nd = dict()
        self.n = dict()
        self.state = self.environment.reset()

        # Resets
        self.reset_iterations()
        self.reset_states_to_observe()

    def best_action(self, state: object = None) -> int:
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

    def track_policy(self, state: object, target: Vector) -> list:
        """
        Runs an episode using one of the learned policies (policy tracking).
        This method tracks a policy with vector-value 'target' from the
        start 'state' until a final state is reached.

        IMPORTANT: When the vectors have not entirely converged yet or the transition scheme is stochastic, the equality
        operator should be relaxed. In this case, the action is to be selected that minimizes the difference between the
        left and the right term. In our experiments, we select the action that minimizes the Manhattan distance between
        these terms.

        :param state: initial state
        :param target: vector value of the policy to follow
        :return:
        """

        path = [state]
        # Flag to indicate that path could be wrong.
        approximate_path = False

        # While s is not terminal
        while not self.environment.is_final(state=state):

            # Path found
            found = False

            # Manhattan_distances
            min_manhattan_distance = float('inf')
            min_q_target = None
            min_action = None

            # For each a in environments actions.
            for a in self.environment.action_space:

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

                        # Update target
                        target = q

                        # Append to path
                        path.append(state)

                        # Path found
                        found = True

                        break

                    # Calc manhattan distance
                    manhattan_distance = um.manhattan_distance(a=v, b=target)

                    # Check if current manhattan distance is lower than min manhattan distance
                    if manhattan_distance < min_manhattan_distance:
                        # Get a copy of action
                        min_action = deepcopy(a)

                        # Get a copy of target
                        min_q_target = deepcopy(q)

                        # Update min manhattan distance
                        min_manhattan_distance = manhattan_distance

            # If path not found
            if not found:
                # This transaction must be determinist s': T(s'|s, a) = 1
                state = self.environment.next_state(action=min_action, state=state)

                # Update target
                target = min_q_target

                # Append to path
                path.append(state)

                # Active approximate path
                approximate_path = True

        if approximate_path:
            print('\033[92m The recovered path could be wrong. \033[0m', end='\n\n')

        return path

    def _best_hypervolume(self) -> float:
        """
        Return best hypervolume for state given.
        :param state:
        :return:
        """

        # Hypervolume list
        hv = list()

        for a in self.environment.action_space:
            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=self.environment.current_state, action=a)

            # Calc hypervolume of Q_set, with reference given.
            hv.append(uh.calc_hypervolume(list_of_vectors=q_set, reference=self.hv_reference))

        return max(hv)

    def hypervolume_evaluation(self, state: object) -> int:
        """
        Calc the hypervolume for each action in state given. (HV-PQL)
        :param state:
        :return:
        """

        actions = list()
        max_evaluation = float('-inf')

        # for each a in actions
        for a in self.environment.action_space:

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

    def cardinality_evaluation(self, state: object) -> int:
        """
        Calc the cardinality for each action in state given. (C-PQL)
        :param state:
        :return:
        """

        # List of all Qs
        all_q = list()

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=state, action=a)

            # for each Q in Q_set(s, a)
            for q in q_set:
                all_q.append(ActionVector(action=a, vector=q))

        # NDQs <- ND(all_q). Keep only the non-dominating solutions
        actions = ActionVector.actions_occurrences_based_m3_with_repetitions(
            vectors=all_q, number_of_actions=action_space.n
        )

        # Get max action
        max_cardinality = max(actions)

        # Get all max actions
        actions = [action for action, cardinality in enumerate(actions) if cardinality == max_cardinality]

        # from best actions get one aleatory
        return self.generator.choice(actions)

    def pareto_evaluation(self, state: object) -> int:
        """
        Calc the pareto for each action in state given. (PO-PQL)
        :param state:
        :return:
        """

        # List of all Qs
        all_q = list()

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=state, action=a)

            # for each Q in Q_set(s, a)
            for q in q_set:
                all_q.append(ActionVector(action=a, vector=q))

        # NDQs <- ND(all_q). Keep only the non-dominating solutions
        actions = ActionVector.actions_occurrences_based_m3_with_repetitions(
            vectors=all_q, number_of_actions=action_space.n
        )

        # Get actions that have almost a non dominated vector.
        actions = [action for action, cardinality in enumerate(actions) if cardinality > 0]

        # If actions is empty, get all available actions.
        if len(actions) <= 0:
            actions = action_space

        # from best actions get one aleatory
        return self.generator.choice(actions)

    def get_dict_model(self) -> dict:
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

    def to_json(self) -> str:
        """
        Get a dict model from current object and return as json string.
        :return:
        """
        model = self.get_dict_model()
        return json.dumps(model, indent=self.json_indent)

    def save(self, filename: str = None) -> None:
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
        file_path = AgentPQL.dumps_file_path(filename=filename)

        # Get dict model
        model = self.get_dict_model()

        # Open file with filename in write mode with UTF-8 encoding.
        with open(file_path, 'w', encoding='UTF-8') as file:
            json.dump(model, file, indent=self.json_indent)

    def non_dominated_vectors_from_state(self, state: object) -> list:
        """
        Return all non dominate vectors from state given.
        :param state:
        :return:
        """

        # Get list of non dominate vectors
        nd_lists = self.nd.get(state, {}).values()

        # Flatten nd_lists
        nd = [item for sublist in nd_lists for item in sublist]

        return self.environment.default_reward.m3_max(nd)

    def print_information(self) -> None:

        super().print_information()

        print("Hypervolume reference: {}".format(self.hv_reference))
        print('Evaluation mechanism: {}'.format(self.evaluation_mechanism))

    @staticmethod
    def load(filename: str = None, environment: Environment = None, evaluation_mechanism: str = None):
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
            environment = um.str_to_snake_case(environment.__class__.__name__)

            # Get evaluation mechanism name in snake case
            evaluation_mechanism = um.str_to_snake_case(evaluation_mechanism)

            # Filter str
            filter_str = '{}_{}'.format(environment, evaluation_mechanism)

            for root, directories, files in os.walk(AgentPQL.dumps_path):

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
        file_path = AgentPQL.dumps_file_path(filename)

        # Read file from path
        try:
            file = open(file_path, 'r', encoding='UTF-8')
        except FileNotFoundError:
            return None

        # Load structured data from indicated file.
        model = json.load(file)

        # Close file
        file.close()

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

            if 'state' in key or 'transitions' in key:
                # Convert to tuples to hash
                value = um.lists_to_tuples(value)

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
            # Convert to tuples to hash
            key = um.lists_to_tuples(item.get('key'))

            value = item.get('value')

            states_to_observe.update({key: value})

        # Unpack 'r' data
        r = dict()
        for item in data.get('r'):
            # Convert to tuples to hash
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
            # Convert to tuples to hash
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
            # Convert to tuples to hash
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
    def dumps_file_path(filename: str) -> str:
        # Return path from file name
        return '{}/{}.json'.format(AgentPQL.dumps_path, filename)
