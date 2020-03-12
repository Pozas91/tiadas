"""
Agent PQL.

Implementation of the PQ-learning algorithm by Kristof Van Moffaert and Ann Nowé, in "Multi-Objective Reinforcement
Learning using Sets of Pareto Dominating Policies" paper.

Save rewards, N and occurrences independently, to calculate Q-set on runtime.

Evaluation mechanisms available
    - HV-PQL: Based on hypervolume
    - C-PQL: Based on cardinality of the set of Pareto-optimal vectors
    - PO-PQL: Based on existence of at least one Pareto-optimal vector
    
    
Sample call: 
    
    1) episode_train
    
        # Instance of environment
        env = DeepSeaTreasure()

        # Instance of agent
        agent = AgentPQL(environment=env)

        # Train agent
        agent.episode_train() # Optional you can pass a number of episodes, e.g. agent.episode_train(episodes=3000)
    
    
    2) write agent to path
    
        # Instance of environment
        env = DeepSeaTreasure()

        # Instance of agent
        agent = AgentPQL(environment=env)

        # Write to path
        agent.save() # Optional you can pass a filename, e.g. agent.save(filename='my_agent')
    
    
    3) read agent from path
    
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
import math
import time
from copy import deepcopy

import numpy as np

import utils.hypervolume as uh
import utils.miscellaneous as um
from agents import Agent
from environments import Environment
from models import IndexVector, GraphType, EvaluationMechanism, Vector, VectorDecimal, AgentType
from .agent_rl import AgentRL


class AgentPQL(AgentRL):

    def __init__(self, environment: Environment, epsilon: float = 0.1, gamma: float = 1., seed: int = 0,
                 max_steps: int = None, hv_reference: Vector = None, graph_types: set = None,
                 evaluation_mechanism: EvaluationMechanism = EvaluationMechanism.HV, initial_value: Vector = None,
                 states_to_observe: set = None):

        """
        :param environment: instance of any environment class.
        :param epsilon: Epsilon used in epsilon-greedy policy, to determine degree of exploration
        :param seed: Seed used for np.random.RandomState method.
        :param max_steps: Limit of steps per episode.
        :param hv_reference: Reference vector for hypervolume calculations
        :param evaluation_mechanism: Evaluation mechanism used to calc best action to choose among those with
               non-dominated policy estimates. Three values are available: EvaluationMechanism.{C, PO, HV}
        :param states_to_observe: List of states to be traced in a graphical output.
        :param graph_types: types of graphical outputs that will be produced.
        """

        # Types to make graphs
        if graph_types is None:
            graph_types = {GraphType.EPISODES, GraphType.STEPS}

        # Super call __init__
        super().__init__(environment=environment, epsilon=epsilon, gamma=gamma, seed=seed,
                         states_to_observe=states_to_observe, max_steps=max_steps, graph_types=graph_types,
                         initial_value=initial_value)

        if initial_value is None:
            self.initial_q_value = self.environment.default_reward.zero_vector
        else:
            self.initial_q_value = initial_value

        # Average observed immediate reward vector.
        self.r = dict()

        # Set of non-dominated vectors
        self.nd = dict()

        # Occurrences
        self.n = dict()

        # Evaluation mechanism
        if evaluation_mechanism in (
                EvaluationMechanism.HV, EvaluationMechanism.PO, EvaluationMechanism.C, EvaluationMechanism.CHV
        ):
            self.evaluation_mechanism = evaluation_mechanism
        else:
            raise ValueError('The evaluation mechanism is not valid.')

        self.hv_reference = hv_reference

    def do_step(self) -> bool:
        # Choose action a from state using a policy derived from the Q-set
        action = self.select_action()

        # perform chosen action on the environment
        next_state, reward, is_final, info = self.environment.step(action=action)

        # Increment steps
        self.total_steps += 1
        self.steps += 1

        # Check if is final
        if is_final:
            # ND(state, a) <- Zero vector
            nd_s_a_dict = self.nd.get(self.state, {})
            nd_s_a_dict.update({action: [self.initial_q_value]})
            self.nd.update({self.state: nd_s_a_dict})
        else:
            # Update ND policies of state' in state
            self.update_nd_s_a(state=self.state, action=action, next_state=next_state)

        # Update numbers of occurrences
        n_s_a = self.get_and_update_n_s_a(state=self.state, action=action)

        # Update average immediate rewards
        self.update_r_s_a(state=self.state, action=action, reward=reward, occurrences=n_s_a)

        # Proceed to next position
        self.state = next_state

        return is_final

    def update_graph(self, graph_type: GraphType) -> None:
        """
        Update specific graph type
        :param graph_type:
        :return:
        """

        if graph_type is GraphType.MEMORY:

            # Count number of vectors in non dominate dictionary
            self.graph_info[graph_type].append(
                sum(len(actions) for states in self.nd.values() for actions in states.values())
            )

        elif graph_type is GraphType.DATA_PER_STATE:

            # Get positions on axis x and y
            x = self.environment.observation_space.spaces[0].n
            y = self.environment.observation_space.spaces[1].n

            # Extract only states with information
            valid_states = self.nd.keys()

            # By default the size of all states is zero
            z = np.zeros([y, x])

            # Calc number of vectors for each position
            for x, y in valid_states:
                z[y][x] = sum(len(actions) for actions in self.nd[(x, y)].values())

            # Save that information
            self.graph_info[graph_type].append(z)

        else:

            # In the same for loop, is check if this agent has the graph_type indicated (get dictionary default
            # value)
            for state, data in self.graph_info.get(graph_type, {}).items():
                # Extract V(position) (without operations)
                value = self.q_set_from_state(state=state)

                # Add information to that train_data
                data.append({
                    'train_data': value,
                    'time': time.time() - self.reference_time_to_train,
                    'iterations': self.total_steps
                })

                # Update dictionary
                self.graph_info[graph_type].update({state: data})

    def get_and_update_n_s_a(self, state: object, action: int) -> int:
        """
        Update n(state, a) dictionary.
        :param state:
        :param action:
        :return:
        """

        # Get n(state) dict
        n_s_a_dict = self.n.get(state, {})
        # Get n(state, a) value
        n_s_a = (n_s_a_dict.get(action, 0) + 1)
        # Update with the increment.
        n_s_a_dict.update({action: n_s_a})
        # Update dictionary
        self.n.update({state: n_s_a_dict})

        return n_s_a

    def update_r_s_a(self, state: object, action: int, reward: Vector, occurrences: int) -> None:
        """
        Update r(state, a) dictionary.
        :param state:
        :param action:
        :param reward:
        :param occurrences:
        :return:
        """

        # Get R(state) dict.
        r_s_a_dict = self.r.get(state, {})
        # Get R(state, a) value.
        r_s_a = r_s_a_dict.get(action, self.environment.default_reward.zero_vector)
        # Update R(state, a)
        r_s_a += (reward - r_s_a) / occurrences
        # Update with new R(state, a)
        r_s_a_dict.update({action: r_s_a})
        # Update R dictionary
        self.r.update({state: r_s_a_dict})

    def update_nd_s_a(self, state: object, action: int, next_state: object) -> None:
        """
        Update ND(state, a)
        :param state:
        :param action:
        :param next_state:
        :return:
        """

        # Union
        union = list()

        # for each action in actions
        for a in self.environment.action_space:
            # Get Q(state, a)
            q = self.q_set(state=next_state, action=a)

            # Union with flatten mode.
            union += q

        # Update ND policies of state' in state
        nd_s_a = self.environment.default_reward.m3_max(union)

        # ND(state, a) <- (ND(U_a' Q_set(state', a'))
        nd_s_a_dict = self.nd.get(state, {})
        nd_s_a_dict.update({action: nd_s_a})
        self.nd.update({state: nd_s_a_dict})

    def q_set(self, state: object, action: int) -> list:
        """
        Calc on run-time Q(state, a)
        :param state:
        :param action:
        :return:
        """

        # Get R(state, a) with default.
        r_s_a = self.r.get(state, {}).get(action, self.environment.default_reward.zero_vector)

        # Get ND(state, a)
        non_dominated_vectors = self.nd.get(state, {}).get(action, [self.initial_q_value])

        # R(state, a) + y*ND
        q_set = [r_s_a + (non_dominated * self.gamma) for non_dominated in non_dominated_vectors]

        return q_set

    def q_set_from_state(self, state: object) -> list:
        """
        Calc Q(state)
        :param state:
        :return:
        """

        # Union
        union = list()

        # for each action in actions
        for a in self.environment.action_space:
            # Get Q(state, a)
            q = self.q_set(state=state, action=a)

            # Union with flatten mode.
            union += q

        # Return Q(state)
        return self.environment.default_reward.m3_max(union)

    def reset(self) -> None:
        """
        Reset agent, forgetting previous dictionaries
        :return:
        """

        # Super call to reset method
        super().reset()

        self.r = dict()
        self.nd = dict()
        self.n = dict()
        self.state = self.environment.reset()

        # Resets
        self.reset_steps()
        self.reset_graph_info()

    def _best_action(self, state: object = None, extra: object = None) -> int:
        """
        Return best action for q and position given. The best action is selected according to the method
        specified in the self.evaluation_mechanism variable.
        :param extra:
        :param state:
        :return:
        """

        # if don't specify a position, get current position.
        if not state:
            state = self.state

        # Use the selected evaluation
        if self.evaluation_mechanism is EvaluationMechanism.HV:
            action = self.hypervolume_evaluation(state=state)
        elif self.evaluation_mechanism is EvaluationMechanism.C:
            action = self.cardinality_evaluation(state=state)
        elif self.evaluation_mechanism is EvaluationMechanism.PO:
            action = self.pareto_evaluation(state=state)
        elif self.evaluation_mechanism is EvaluationMechanism.CHV:
            action = self.chv_evaluation(state=state)
        else:
            raise ValueError('Unknown evaluation mechanism')

        return action

    def track_policy(self, state: object, target: Vector) -> list:
        """
        Runs an episode using one of the learned policies (policy tracking).
        This method tracks a policy with vector-value 'target' from the
        start 'position' until a final position is reached.

        IMPORTANT: When the vectors have not entirely converged yet or the transition scheme is stochastic, the equality
        operator should be relaxed. In this case, the action is to be selected that minimizes the difference between the
        left and the right term. In our experiments, we select the action that minimizes the Manhattan distance between
        these terms.

        :param state: initial position
        :param target: vector value of the policy to follow
        :return:
        """

        path = [state]
        # Flag to indicate that path could be wrong.
        approximate_path = False

        # While state is not terminal
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

                # Retrieve R(state, a)
                r = self.r.get(state).get(a)

                # Retrieve ND(state, a)
                nd = self.nd.get(state).get(a)

                for q in nd:

                    # Calc vector to follow
                    v = (q * self.gamma) + r

                    # if are equals with relaxed equality operator
                    if v.all_close(target):
                        # This transaction must be determinist state': T(state'|state, a) = 1
                        state = self.environment.next_state(action=a, position=state)

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
                # This transaction must be determinist state': T(state'|state, a) = 1
                state = self.environment.next_state(action=min_action, position=state)

                # Update target
                target = min_q_target

                # Append to path
                path.append(state)

                # Active approximate path
                approximate_path = True

        if approximate_path:
            print('\033[92m The recovered path could be wrong. \033[0m', end='\n\n')

        return path

    def _best_hypervolume(self, state: object = None) -> float:
        """
        Return best hypervolume for position given.
        :return:
        """

        # Check if a position is given
        state = state if state else self.environment.current_state

        # Hypervolume list
        hv = list()

        # Save previous position
        previous_state = self.environment.current_state
        self.environment.current_state = state

        for a in self.environment.action_space:
            # Get Q-set from position given for each possible action.
            q_set = self.q_set(state=state, action=a)

            # Calc hypervolume of Q_set, with reference given.
            hv.append(uh.calc_hypervolume(vectors=q_set, reference=self.hv_reference))

        # Restore environment correct position
        self.environment.current_state = previous_state

        return max(hv)

    def hypervolume_evaluation(self, state: object) -> int:
        """
        Calc the hypervolume for each action in the given position, and returns the int representing the action
        with maximum hypervolume. (Approximate) ties are broken choosing randomly among actions with
        (approximately) maximum hypervolume. (EvaluationMechanism.HV)
        :param state:
        :return:
        """

        actions = list()
        max_evaluation = float('-inf')

        # for each a in actions
        for a in self.environment.action_space:

            # Get Q-set from position given for each possible action.
            q_set = self.q_set(state=state, action=a)

            # Calc hypervolume of Q_set, with reference given.
            evaluation = uh.calc_hypervolume(vectors=q_set, reference=self.hv_reference)

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
        Calculates the cardinality (number of Pareto-optimal estimates) for each action in the given position,
        and returns the int representing an action with maximum cardinality. Ties are broken randomly among
        max cardinality actions.
        (EvaluationMechanism.C)
        :param state:
        :return:
        """

        # List of all Qs
        all_q = list()

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from position given for each possible action.
            q_set = self.q_set(state=state, action=a)

            # for each Q in Q_set(state, a)
            for q in q_set:
                all_q.append(IndexVector(index=a, vector=q))

        # NDQs <- ND(all_q). Keep only the non-dominating solutions
        actions = IndexVector.actions_occurrences_based_m3_with_repetitions(
            vectors=all_q, actions=action_space
        )

        # Get max action
        max_cardinality = max(actions.values())

        # Get all max actions
        filter_actions = [action for action in actions.keys() if actions[action] == max_cardinality]

        # choose randomly among actions with maximum cardinality
        return self.generator.choice(filter_actions)

    def chv_evaluation(self, state: object) -> int:
        """
        Calc the hypervolume for the vectors that provide cardinality for each action and returns a tuple
        (maximum_chv, list of tuples (action, chv), sum of chv)

        CAUTION: This method assumes actions are integers in a range.

        :param state:
        :return:
        """

        # List of all Qs
        all_q = list()

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from position given for each possible action.
            q_set = self.q_set(state=state, action=a)

            # for each Q in Q_set(state, a)
            for q in q_set:
                all_q.append(IndexVector(index=a, vector=q))

        # NDQs <- ND(all_q). Keep only the non-dominating solutions (We want the vectors, so return_vectors must be
        # True)
        vectors_dict = IndexVector.actions_occurrences_based_m3_with_repetitions(
            vectors=all_q, actions=action_space, returns_vectors=True
        )

        # Dict where each action has it hypervolume
        hypervolume_actions = {
            action: uh.calc_hypervolume(vectors=vectors, reference=self.hv_reference) if len(vectors) > 0
            else 0.0
            for action, vectors in vectors_dict.items()
        }

        # Get max hypervolume
        max_hypervolume = max(hypervolume_actions.values())

        # Get all max actions
        filter_actions = [
            action for action in hypervolume_actions.keys() if hypervolume_actions[action] == max_hypervolume
        ]

        # Choose randomly among actions with maximum hypervolume
        return self.generator.choice(filter_actions)

    def pareto_evaluation(self, state: object) -> int:
        """
        Calculates which actions for the current position have at least a Pareto-optimal estimate, and returns one
         of them (randomly chosen). (EvaluationMechanism.PO)
        :param state:
        :return:
        """

        # List of all Qs
        all_q = list()

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from position given for each possible action.
            q_set = self.q_set(state=state, action=a)

            # for each Q in Q_set(state, a)
            for q in q_set:
                all_q.append(IndexVector(index=a, vector=q))

        # NDQs <- ND(all_q). Keep only the non-dominating solutions
        actions = IndexVector.actions_occurrences_based_m3_with_repetitions(
            vectors=all_q, actions=action_space
        )

        # Get all max actions
        filter_actions = [action for action in actions.keys() if actions[action] > 0]

        # If actions is empty, get all available actions.
        if len(filter_actions) <= 0:
            filter_actions = action_space

        # from best actions get one aleatory
        return self.generator.choice(filter_actions)

    def get_dict_model(self) -> dict:
        """
        Get a dictionary of model
        In JSON serialize only is valid strings as key on dict, so we convert all numeric keys in strings keys.
        :return:
        """

        # Get parent'state model
        model = super().get_dict_model()

        # Own properties
        model['train_data'].update({
            'r': [
                {
                    'key': k, 'value': {'key': int(k2), 'value': v2.tolist()}
                } for k, v in self.r.items() for k2, v2 in v.items()
            ]
        })

        model['train_data'].update({
            'nd': [
                {
                    'key': k, 'value': {'key': int(k2), 'value': [vector.tolist() for vector in v2]}
                } for k, v in self.nd.items() for k2, v2 in v.items()
            ]
        })

        model['train_data'].update({
            'n': [
                {
                    'key': k, 'value': {'key': int(k2), 'value': v2}
                } for k, v in self.n.items() for k2, v2 in v.items()
            ]
        })

        model['train_data'].update({'hv_reference': self.hv_reference.tolist()})
        model['train_data'].update({'evaluation_mechanism': str(self.evaluation_mechanism)})
        model['train_data'].update({'total_episodes': self.total_episodes})
        model['train_data'].update({'total_steps': self.total_steps})
        model['train_data'].update({'position': list(self.state)})

        return model

    def non_dominated_vectors_from_state(self, state: object) -> list:
        """
        Return all non dominate vectors from position given.
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

    def json_filename(self) -> str:
        """
        Generate a filename for json save path
        :return:
        """
        # Get environment name in snake case
        environment = um.str_to_snake_case(self.environment.__class__.__name__)

        # Get evaluation mechanism in snake case
        agent = um.str_to_snake_case(self.__class__.__name__)

        # Get evaluation mechanism in snake case
        evaluation_mechanism = um.str_to_snake_case(str(self.evaluation_mechanism.value))

        # Get date
        date = datetime.datetime.now().timestamp()

        return '{}_{}_{}_{}'.format(agent, environment, evaluation_mechanism, date)

    @staticmethod
    def load(filename: str = None, **kwargs) -> object:

        # Load structured train_data from indicated path.
        model = Agent.load(filename=filename, agent_type=AgentType.PQL)

        # Get meta-train_data
        meta = model.get('meta')
        # Get train_data
        model_data = model.get('train_data')

        # ENVIRONMENT
        environment = model_data.get('environment')

        # Meta
        environment_meta = environment.get('meta')
        environment_class_name = environment_meta.get('class')
        environment_module_name = environment_meta.get('module')
        environment_module = importlib.import_module(environment_module_name)
        environment_class_ = getattr(environment_module, environment_class_name)

        # Data
        environment_data = environment.get('train_data')

        # Instance
        environment = environment_class_()

        # Set environment train_data
        for key, value in environment_data.items():

            if 'position' in key or 'p_stochastic' in key:
                # Convert to tuples to hash
                value = um.lists_to_tuples(value)

            elif 'default_reward' in key:
                # If all elements are int, then default_reward is a integer Vector, otherwise float Vector
                value = Vector(value) if (all([isinstance(x, int) for x in value])) else VectorDecimal(value)

            vars(environment)[key] = value

        # Set initial_seed
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
        epsilon = model_data.get('epsilon')
        gamma = model_data.get('gamma')
        total_episodes = model_data.get('total_episodes')
        total_steps = model_data.get('total_steps')
        state = tuple(model_data.get('position'))
        max_steps = model_data.get('max_steps')
        seed = model_data.get('initial_seed')

        # Recover evaluation mechanism from string
        evaluation_mechanism = EvaluationMechanism.from_string(
            evaluation_mechanism=model_data.get('evaluation_mechanism'))

        # default_reward is reference so, reset components (multiply by zero) and add hv_reference to get hv_reference.
        hv_reference = default_reward.zero_vector + model_data.get('hv_reference')

        # Prepare Graph Types
        graph_types = set()

        # Update 'states_to_observe' train_data
        states_to_observe = dict()
        for item in model_data.get('states_to_observe'):

            # Get graph type
            key = GraphType.from_string(item.get('key'))
            graph_types.add(key)

            value = item.get('value')
            state = um.lists_to_tuples(value.get('key'))
            value = value.get('value')

            if key not in states_to_observe.keys():
                states_to_observe.update({
                    key: dict()
                })

            states_to_observe.get(key).update({
                state: value
            })

        # Unpack 'r' train_data
        r = dict()
        for item in model_data.get('r'):
            # Convert to tuples to hash
            key = um.lists_to_tuples(item.get('key'))
            value = item.get('value')
            action = value.get('key')
            value = default_reward.zero_vector + value.get('value')

            if key not in r.keys():
                r.update({key: dict()})

            r.get(key).update({action: value})

        # Unpack 'nd' train_data
        nd = dict()
        for item in model_data.get('nd'):
            # Convert to tuples to hash
            key = um.lists_to_tuples(item.get('key'))
            value = item.get('value')
            action = value.get('key')
            value = [default_reward.zero_vector + v for v in value.get('value')]

            if key not in nd.keys():
                nd.update({key: dict()})

            nd.get(key).update({action: value})

        # Unpack 'n' train_data
        n = dict()
        for item in model_data.get('n'):
            # Convert to tuples to hash
            key = um.lists_to_tuples(item.get('key'))
            value = item.get('value')
            action = value.get('key')
            value = int(value.get('value'))

            if key not in n.keys():
                n.update({key: dict()})

            n.get(key).update({action: value})

        # Prepare an instance of model.
        model = AgentPQL(environment=environment, epsilon=epsilon, gamma=gamma, seed=seed, max_steps=max_steps,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, graph_types=graph_types)

        # Set finals settings and return it.
        model.r = r
        model.nd = nd
        model.n = n
        model.graph_info = states_to_observe
        model.state = state
        model.total_episodes = total_episodes
        model.total_steps = total_steps

        return model

    def objective_training(self, list_of_vectors: list, graph_type: GraphType = None):
        """
        Train until V(s0) value is close to objective value.
        :param graph_type:
        :param list_of_vectors:
        :return:
        """

        # Calc current hypervolume
        current_hypervolume = self._best_hypervolume(self.environment.initial_state)

        objective_hypervolume = uh.calc_hypervolume(vectors=list_of_vectors, reference=self.hv_reference)

        while not math.isclose(a=current_hypervolume, b=objective_hypervolume, rel_tol=0.01, abs_tol=0.0):
            # Do an episode
            self.episode(graph_type=graph_type)

            # Update hypervolume
            current_hypervolume = self._best_hypervolume(self.environment.initial_state)

        pass
