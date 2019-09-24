"""
First version of algorithm 1.
Adaptation of algorithm MPQ-learning to directed acyclic graphs.
"""
import datetime
import importlib
import itertools
import json
import math
import os
import time
from copy import deepcopy

import gym
import numpy as np

import utils.hypervolume as uh
import utils.miscellaneous as um
from agents import Agent
from environments import Environment
from models import Vector, IndexVector, VectorDecimal, GraphType, EvaluationMechanism


class AgentA1(Agent):

    def __init__(self, environment: Environment, hv_reference: Vector, alpha: float = 0.1, epsilon: float = 0.1,
                 gamma: float = 1., seed: int = 0, states_to_observe: list = None, max_steps: int = None,
                 evaluation_mechanism: EvaluationMechanism = EvaluationMechanism.HV,
                 graph_types: set = None, integer_mode: bool = True):
        """
        :param environment: An environment where agent does any operation.
        :param alpha: Learning rate
        :param epsilon: Epsilon using in e-greedy policy, to explore more states.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param states_to_observe: List of states from that we want to get a graphical output.
        :param max_steps: Limits of steps per episode.
        :param hv_reference: Reference vector to calc hypervolume
        :param evaluation_mechanism: Evaluation mechanism used to calc best action to choose. Three values are
            available: 'C-PQL', 'PO-PQL', 'HV-PQL'
        :param graph_types: Set of types of graph to generate.
        """

        # Types to show a graphs
        if graph_types is None:
            graph_types = {GraphType.STEPS, GraphType.MEMORY}

        # Super call __init__
        super().__init__(environment=environment, epsilon=epsilon, gamma=gamma, seed=seed,
                         states_to_observe=states_to_observe, max_steps=max_steps, graph_types=graph_types)

        # Learning factor
        assert 0 < alpha <= 1
        self.alpha = alpha

        # Dictionary that stores all q values. 
        #Key: state; Value: second level dictionary.
        #Second level dictionary: key: action; value: third level dictionary
        #Third level dictionary: key :index vector (element from cartesian product); 
        #                        value: q-vector (instance of class IndexVector)
        self.q = dict()

        # States known by each state and action
        self.s = dict()

        # Return non dominate states for a state given
        self.v = dict()

        # Counter to indexes used by each pair (state, action)
        self.indexes_counter = dict()

        # Set of states that need be updated
        self.states_to_update = set()

        # Evaluation mechanism
        if evaluation_mechanism in (EvaluationMechanism.HV, EvaluationMechanism.PO, EvaluationMechanism.C):
            self.evaluation_mechanism = evaluation_mechanism
        else:
            raise ValueError('Evaluation mechanism does not valid.')

        # if integer mode is True, all reward vectors received will be converted.
        self.integer_mode = integer_mode

        # Hypervolume reference
        if self.integer_mode:
            hv_reference = hv_reference.to_decimals()

        self.hv_reference = hv_reference

    def get_dict_model(self) -> dict:
        """
        Get a dictionary of model
        In JSON serialize only is valid strings as key on dict, so we convert all numeric keys in strings keys.
        :return:
        """

        # Get parent's model
        model = super().get_dict_model()

        # Own properties
        model.get('data').update({
            'indexes_counter': [
                {
                    'key': list(k), 'value': v
                } for k, v in self.indexes_counter.items()
            ]
        })

        model.get('data').update({
            'q': [
                dict(key=list(state), value={
                    'key': int(action), 'value': {
                        'key': list(table_index), 'value': (v3.index, v3.vector.tolist())
                    }
                }) for state, v in self.q.items() for action, v2 in v.items() for table_index, v3
                in v2.items()
            ]
        })

        model.get('data').update({
            's': [
                dict(key=list(state), value={
                    'key': int(action), 'value': v2
                }) for state, v in self.s.items() for action, v2 in v.items()
            ]
        })

        model.get('data').update({
            'v': [
                dict(key=list(state), value={
                    'key': int(vector_index), 'value': v2.tolist()
                }) for state, v in self.v.items() for vector_index, v2 in v.items()
            ]
        })

        model.get('data').update({'alpha': self.alpha})
        model.get('data').update({'hv_reference': self.hv_reference.tolist()})
        model.get('data').update({'evaluation_mechanism': str(self.evaluation_mechanism)})
        model.get('data').update({'integer_mode': self.integer_mode})
        model.get('data').update({'total_episodes': self.total_episodes})
        model.get('data').update({'total_steps': self.total_steps})
        model.get('data').update({'state': list(self.state)})

        return model

    def do_iteration(self) -> bool:

        # If the state is unknown, register it.
        if self.state not in self.q:
            self.q.update({self.state: dict()})

        if self.state not in self.s:
            self.s.update({self.state: dict()})

        if self.state not in self.v:
            self.v.update({self.state: dict()})

        if self.state not in self.indexes_counter:
            # Initialize counters
            self.indexes_counter.update({self.state: 0})

        # Get an action
        action = self.select_action()

        # Do step on environment
        next_state, reward, is_final_state, info = self.environment.step(action=action)

        if self.integer_mode:
            reward = reward.to_decimals()

        # Increment steps done
        self.total_steps += 1
        self.steps += 1

        # If next_state is a final state and not is register
        if is_final_state:

            # If not is register in V, register it
            if not self.v.get(next_state):
                self.v.update({
                    next_state: {
                        # By default finals states has a zero vector with a zero index
                        0: self.environment.default_reward.zero_vector
                    }
                })

        # S(s) -> All known states with its action for the state given.
        pair_action_states_known_by_state = self.s.get(self.state)

        # S(s, a) -> All known states for state and action given.
        states_known_by_state = pair_action_states_known_by_state.get(action, list())

        # I_s_k
        relevant_indexes_of_next_state = self.relevant_indexes_of_state(state=next_state)

        # S_k in S_{n - 1}
        next_state_is_in_states_known = next_state in states_known_by_state

        # Check if sk not in S, and I_s_k is not empty
        if not next_state_is_in_states_known and relevant_indexes_of_next_state:
            # Q_n = N_n(s, a)
            self.new_operation(state=self.state, action=action, reward=reward, next_state=next_state)

        elif next_state_is_in_states_known:
            # Q_n = U_n(s, a)
            self.update_operation(state=self.state, action=action, reward=reward, next_state=next_state)

        # Check if is necessary update V(s) to improve the performance
        self.check_if_need_update_v()

        # Update state
        self.state = next_state

        return is_final_state

    def update_graph(self, graph_types: tuple) -> None:
        """
        Update specific graph type
        :param graph_types:
        :return:
        """

        for graph_type in graph_types:

            # Check if that type of graph is in our agent
            if graph_type in self.graph_info:

                if graph_type is GraphType.MEMORY:

                    # Count number of vectors in big Q dictionary
                    self.graph_info[graph_type].append(
                        sum(len(actions.values()) for states in self.q.values() for actions in states.values())
                    )

                elif graph_type is GraphType.VECTORS_PER_CELL:

                    # Get positions on axis x and y
                    x = self.environment.observation_space.spaces[0].n
                    y = self.environment.observation_space.spaces[1].n

                    # Extract only states with information
                    valid_states = self.q.keys()

                    # By default the size of all states is zero
                    z = np.zeros([y, x])

                    # Calc number of vectors for each state
                    for x, y in valid_states:
                        z[y][x] = sum(len(actions.values()) for actions in self.q[(x, y)].values())

                    # Save that information
                    self.graph_info[graph_type].append(z)

                else:

                    # In the same for loop, is check if this agent has the graph_type indicated (get dictionary default
                    # value)
                    for state, data in self.graph_info.get(graph_type, {}).items():

                        # Add to data Best value (V max)
                        value = self._best_hypervolume(state=state)

                        # If integer mode is True, is necessary divide value by increment
                        if self.integer_mode:
                            # Divide value by two powered numbers (hv_reference and reward)
                            value /= 10 ** (Vector.decimals_allowed * 2)

                        # Add to data Best value (V max)
                        data.append(value)

                        # Update dictionary
                        self.graph_info[graph_type].update({state: data})

                    if graph_type is GraphType.TIME:
                        # Update last execution
                        self.last_time_to_get_graph_data = time.time()

    def _best_hypervolume(self, state: object = None) -> float:
        """
        Return best hypervolume for state given.
        :return:
        """

        # Check if a state is given
        state = state if state else self.environment.current_state

        # Get Q-set from state given for each possible action
        v = list(self.v.get(state, {}).values())

        # If v is empty, default is zero vector
        v = v if v else [self.environment.default_reward.zero_vector]

        # Getting hypervolume
        hv = uh.calc_hypervolume(list_of_vectors=v, reference=self.hv_reference)

        return hv

    def reverse_episode(self, episodes_per_state: int = 10) -> None:
        """
        Run an episode complete until get a final step
        :return:
        """

        objective_states = set(self.environment.finals.keys())
        invalid_states = objective_states.union(self.environment.obstacles)

        visited_states = {key: set() for key in objective_states}

        counter_visited_states = dict()

        more_valid_states = True
        distance = 1

        while more_valid_states:

            # Neighbours by objective state
            all_neighbours = dict()

            # Get all neighbours from known objectives
            for state in objective_states:
                # Getting nearest neighbours (removing visited_states)
                neighbours = set(filter(
                    lambda x: self.environment.observation_space.contains(x) and x not in invalid_states,
                    self.calc_neighbours(from_state=state, distance=distance)
                )) - visited_states.get(state)

                all_neighbours.update({state: neighbours})

            # Getting total states to visit
            states_to_visit = set().union(*all_neighbours.values())
            number_of_states_to_visit = len(states_to_visit)

            # Check if there are more valid states pending
            more_valid_states = number_of_states_to_visit > 0

            # Number of states to visit per episodes per state
            total_episodes = number_of_states_to_visit * episodes_per_state

            for episode in range(total_episodes):

                if not states_to_visit:
                    print('Not more states to visit. Stop in episode {}'.format(episode))
                    break

                # Get possible combinations
                possible_neighbours = filter(lambda x: len(x[1]) > 0, all_neighbours.items())

                # Get random state
                objective_state, associate_states = self.generator.choice(tuple(possible_neighbours))
                selected_state = self.generator.choice(tuple(associate_states))

                # Interesting counter
                if selected_state not in counter_visited_states:
                    counter_visited_states.update({
                        selected_state: 1
                    })
                else:
                    counter_visited_states.update({
                        selected_state: counter_visited_states.get(selected_state) + 1
                    })

                # Change initial state for train
                self.environment.initial_state = selected_state

                # Do an episode
                self.episode()

                # Do while any objective has any state not empty
                states_to_visit = any([x for x in all_neighbours.values()])

            distance += 1

    def best_action(self, state: object = None) -> int:
        """
        Return best action a state given.
        :return:
        """

        # if don't specify a state, get current state.
        if not state:
            state = self.state

        # Use the selected evaluation
        if self.evaluation_mechanism == EvaluationMechanism.HV:
            action = self.hypervolume_evaluation(state=state)
        elif self.evaluation_mechanism == EvaluationMechanism.C:
            action = self.cardinality_evaluation(state=state)
        else:
            action = self.pareto_evaluation(state=state)

        return action

    def hypervolume_evaluation(self, state: object) -> int:
        """
        Calc the hypervolume for each action in state given. (HV-PQL)
        :param state:
        :return:
        """

        actions = list()
        max_evaluation = float('-inf')

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from state given for each possible action
            q_set = self.q.get(state, dict()).get(a, {
                (0,): IndexVector(index=0, vector=self.environment.default_reward.zero_vector)
            })

            # Filter vector from index vectors
            q_set = [q.vector for q in q_set.values()]

            # Calc hypervolume of Q_set, with reference given
            evaluation = uh.calc_hypervolume(list_of_vectors=q_set, reference=self.hv_reference)

            # If current value is close to new value
            if math.isclose(a=evaluation, b=max_evaluation):
                # Append another possible action
                actions.append(a)

            elif evaluation > max_evaluation:
                # Create a new list with current key
                actions = [a]

            # Update max value
            max_evaluation = max(max_evaluation, evaluation)

        # from best actions get one randomly
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

            # Get Q-set from state given for each possible action
            q_set = self.q.get(state, dict()).get(a, {
                (0,): IndexVector(index=a, vector=self.environment.default_reward.zero_vector)
            })

            # for each Q in Q_set(s, a)
            for q in q_set.values():
                all_q.append(IndexVector(index=a, vector=q.vector))

        # NDQs <- ND(all_q). Keep only the non-dominating solutions
        actions = IndexVector.actions_occurrences_based_m3_with_repetitions(
            vectors=all_q, actions=action_space
        )

        # Get max action
        max_cardinality = max(actions.values())

        # Get all max actions
        filter_actions = [action for action in actions.keys() if actions[action] == max_cardinality]

        # from best actions get one aleatory
        return self.generator.choice(filter_actions)

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

            # Get Q-set from state given for each possible action
            q_set = self.q.get(state, dict()).get(a, {
                (0,): IndexVector(index=a, vector=self.environment.default_reward.zero_vector)
            })

            # for each Q in Q_set(s, a)
            for q in q_set.values():
                all_q.append(IndexVector(index=a, vector=q.vector))

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

    def new_operation(self, state: object, action: int, reward: Vector, next_state: object) -> None:
        """
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :return:
        """

        # Get states with its actions
        states_with_actions = self.s.get(state)

        # Prepare to add next_state to known states.
        states = states_with_actions.get(action, list())
        states.append(next_state)
        states_with_actions.update({action: states})

        # Save index of next_state (sk)
        next_state_position = states.index(next_state)

        # Calculate the cartesian product
        cartesian_product, indexes = self.cartesian_product_of_relevant_indexes(states=states)

        # Prepare V to do queries
        v = self.v.get(next_state)

        # Index counter
        index_counter = self.indexes_counter.get(state)

        # Data state Q(s)
        data_state = self.q.get(state)

        # Data action Q(s, a)
        data_action = data_state.get(action, dict())

        # Control update flag
        need_update_v = False

        for p_prime in cartesian_product:

            # p is p_prime less index of next_state (p = p' / sk)
            p = p_prime[:next_state_position] + p_prime[next_state_position + 1:]

            # p[s_k]
            next_state_index = p_prime[next_state_position]

            # alpha * (reward + gamma * associate_vector)
            next_q = (reward + v.get(next_state_index) * self.gamma) * self.alpha

            # Q_{n - 1}(s, a, p)
            previous_q = data_state.get(action, {}).get(p)

            # If not exists Q_{n - 1}(s, a, p), then create a new IndexVector with the next index available.
            if not previous_q:
                # Q_{n - 1}(s, a, p)
                previous_q = self.default_vector_value(index=index_counter)

                # Update index counter
                index_counter += 1
                self.indexes_counter[state] = index_counter

            # (1 - alpha) * Q_{n - 1}(s, a, p)
            previous_q = previous_q * (1 - self.alpha)

            # Q(s, a)
            q = IndexVector(index=previous_q.index, vector=previous_q.vector + next_q)

            # Q(s, a, p_prime)
            data_action.update({p_prime: q})

            # Need delete previous index of table if is necessary
            if p != p_prime:
                data_action.pop(p, None)

            data_state.update({action: data_action})
            need_update_v = True

        # Check if is necessary update V vectors
        if need_update_v:
            self.states_to_update.add(state)

    def update_operation(self, state: object, action: int, reward: Vector, next_state: object) -> None:
        """
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :return:
        """

        # Prepare to add next_state to known states
        known_states = self.s.get(state).get(action, list())

        # Save index of next_state (sk)
        next_state_p_position = known_states.index(next_state)

        # Calculate the cartesian product
        cartesian_product, relevant_indexes = self.cartesian_product_of_relevant_indexes(states=known_states)

        # Prepare V to do queries
        v = self.v.get(next_state)

        # Index counter
        index_counter = self.indexes_counter.get(state)

        # Data state Q(s)
        data_state = self.q.get(state)

        # Data action state Q(s, a)
        data_action = data_state.get(action, dict())

        # Control update flag
        need_update_v = False

        # Control of positions used updates to remove orphans vectors
        positions_updated = set()

        # For each tuple in cartesian product
        for p in cartesian_product:

            # Position updated
            positions_updated.add(p)

            # p[s_k]
            next_state_reference_vector_index = p[next_state_p_position]

            # alpha * (reward + gamma * associate_vector)
            next_q = (reward + v.get(next_state_reference_vector_index) * self.gamma) * self.alpha

            # Q_{n - 1}(s, a, p)
            previous_q = data_state.get(action, {}).get(p)

            # If not exists Q_{n - 1}(s, a, p), then create a new IndexVector with the next index available.
            if not previous_q:
                # Q_{n - 1}(s, a, p)
                previous_q = self.default_vector_value(index_counter)

                # Update index counter
                index_counter += 1
                self.indexes_counter[state] = index_counter

            # (1 - alpha) * Q_{n - 1}(s, a, p)
            previous_q = previous_q * (1 - self.alpha)

            # Q(s, a)
            q = IndexVector(index=previous_q.index, vector=previous_q.vector + next_q)

            # Q(s, a, p)
            data_action.update({p: q})
            data_state.update({action: data_action})

            # Is necessary update v vector
            need_update_v = True

        # Get positions from table
        all_positions = set(data_action.keys())

        # Deleting difference between all_positions and positions_updated to del orphans vectors
        for position in all_positions - positions_updated:
            # Removing orphan vector
            del data_state[action][position]

        # Check if is necessary update V vectors
        if need_update_v:
            self.states_to_update.add(state)

    def default_vector_value(self, index):
        """
        This method get a default value if a vector doesn't exist. It's possible change the zero vector for a
        heuristic function.
        :param index:
        :return:
        """
        return IndexVector(index=index, vector=self.environment.default_reward.zero_vector)

    def check_if_need_update_v(self) -> None:

        # For each state that need be updated
        for state in self.states_to_update:
            # Get Q(s)
            q_s = self.q[state]

            # List accumulative to save all vectors
            q_list = list()

            # Save previous state
            previous_state = self.environment.current_state

            # Set new state
            self.environment.current_state = state

            # for each action available in state given
            for a in self.environment.action_space:
                # Q(s, a)
                q_list += q_s.get(a, dict()).values()

            # Get all non dominated vectors -> V(s)
            non_dominated_vectors, _ = self.environment.default_reward.m3_max_2_lists_with_buckets(vectors=q_list)

            v = dict()

            # Sort keys of buckets
            for bucket in non_dominated_vectors:
                bucket.sort(key=lambda x: x.index)

                # Set V values (Getting vector with lower index) (simplified)
                v.update({bucket[0].index: bucket[0].vector})

            # Update V(s)
            self.v.update({state: v})

            # Restore state
            self.environment.current_state = previous_state

        # All pending states are updated
        self.states_to_update.clear()

    def relevant_indexes_of_state(self, state: object) -> set:
        """
        Return a set of relevant indexes from V(s)
        :param state:
        :return:
        """
        return set(self.v.get(state, {}).keys())

    def cartesian_product_of_relevant_indexes(self, states: list):
        """
        :param states:
        :return:
        """

        # Getting relevant indexes for each state
        indexes = [self.relevant_indexes_of_state(state=iter_state) for iter_state in states]

        # Cartesian product of that indexes
        cartesian_product = itertools.product(*indexes)

        return cartesian_product, indexes

    def q_real(self):
        """
        Return a real values of Q-table
        :return:
        """

        # First do a deepcopy of original Q-table
        q = deepcopy(self.q)

        # For each state with it action dictionary
        for state, action_dict in q.items():

            # For each action with it indexes dictionary
            for action, indexes_dict in action_dict.items():
                # For each index with it index_vector (with a int vector associated multiplied by Vector.decimals)
                for index, index_vector in indexes_dict.items():
                    # Divide that vector by Vector.decimals to convert in original float vector
                    index_vector.vector = VectorDecimal(
                        index_vector.vector.components / (10 ** Vector.decimals_allowed)
                    )

                    # Update Q-table dictionary
                    indexes_dict.update({index: index_vector})
                    action_dict.update({action: indexes_dict})
                    q.update({state: action_dict})

        # Return Q-table transformed
        return q

    def v_real(self) -> dict:
        """
        Return a real values of V-values
        :return:
        """

        # First do a deepcopy of original V-values
        v = deepcopy(self.v)

        # For each state with it vectors
        for state, vectors in v.items():

            # For each index with it vector
            for index, vector in vectors.items():
                # Divide that vector by Vector.decimals to convert in original float vector
                vectors.update({
                    index: VectorDecimal(
                        vector.components / (10 ** Vector.decimals_allowed)
                    )
                })
                # Update V-values dictionary
                v.update({state: vectors})

        return v

    def objective_training(self, list_of_vectors: list):
        """
        Train until agent V(0, 0) value is close to objective value.
        :param list_of_vectors:
        :return:
        """

        # Calc current hypervolume
        current_hypervolume = self._best_hypervolume(self.environment.initial_state)

        objective_hypervolume = uh.calc_hypervolume(list_of_vectors=list_of_vectors, reference=self.hv_reference)

        while not math.isclose(a=current_hypervolume, b=objective_hypervolume, rel_tol=0.02):
            # Do an episode
            self.episode()

            # Update hypervolume
            current_hypervolume = self._best_hypervolume(self.environment.initial_state)

    def json_filename(self) -> str:
        """
        Generate a filename for json dump file
        :return:
        """
        # Get environment name in snake case
        environment = um.str_to_snake_case(self.environment.__class__.__name__)

        # Get evaluation mechanism in snake case
        agent = um.str_to_snake_case(self.__class__.__name__)

        # Get evaluation mechanism in snake case
        evaluation_mechanism = um.str_to_snake_case(str(self.evaluation_mechanism))

        # Get date
        date = datetime.datetime.now().timestamp()

        return '{}_{}_{}_{}'.format(agent, environment, evaluation_mechanism, date)

    def initialize_dictionaries(self):
        """
        Function to initialize all possible states in this problem
        :return:
        """
        states = list()

        if isinstance(self.environment.observation_space, gym.spaces.Tuple):

            for x in range(self.environment.observation_space.spaces[0].n):
                for y in range(self.environment.observation_space.spaces[1].n):
                    states.append((x, y))

        elif isinstance(self.environment.observation_space, gym.spaces.Discrete):
            states.append(range(self.environment.observation_space.n))

        for state in states:
            self.s.update({state: dict()})
            self.v.update({state: dict()})
            self.indexes_counter.update({state: 0})

    @staticmethod
    def load(filename: str = None, environment: Environment = None, evaluation_mechanism: EvaluationMechanism = None):
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
            evaluation_mechanism_value = um.str_to_snake_case(str(evaluation_mechanism.value))

            # Filter str
            filter_str = '{}_{}'.format(environment, evaluation_mechanism_value)

            # Filter files with that environment and evaluation mechanism
            files = filter(lambda f: filter_str in f,
                           [path.name for path in os.scandir(AgentA1.dumps_path) if path.is_file()])

            # Files to list
            files = list(files)

            # Sort list of files
            files.sort()

            # At least must have a file
            if files:
                # Get last filename
                filename = files[-1]

        # Prepare file path
        file_path = AgentA1.dumps_file_path(filename)

        # Read file from path
        try:
            file = file_path.open(mode='r', encoding='UTF-8')
        except FileNotFoundError:
            return None

        # Load structured data from indicated file.
        model = json.load(file)

        # Close file
        file.close()

        # Get meta-data
        # model_meta = model.get('meta')

        # Get data
        model_data = model.get('data')

        # ENVIRONMENT
        environment = model_data.get('environment')

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
                value = Vector(value) if (all([isinstance(x, int) for x in value])) else VectorDecimal(value)

            vars(environment)[key] = value

        # Set seed
        environment.seed(seed=environment.initial_seed)

        # Get default reward as reference
        default_reward = environment.default_reward

        # AgentA1

        # Meta

        # Prepare module and class to make an instance.
        # class_name = model_meta.get('class')
        # module_name = model_meta.get('module')
        # module = importlib.import_module(module_name)
        # class_ = getattr(module, class_name)

        # Data
        epsilon = model_data.get('epsilon')
        gamma = model_data.get('gamma')
        alpha = model_data.get('alpha')
        integer_mode = model_data.get('integer_mode')
        total_episodes = model_data.get('total_episodes')
        total_steps = model_data.get('total_steps')
        state = tuple(model_data.get('state'))
        max_steps = model_data.get('max_steps')
        seed = model_data.get('seed')

        # Recover evaluation mechanism from string
        evaluation_mechanism = EvaluationMechanism.from_string(
            evaluation_mechanism=model_data.get('evaluation_mechanism'))

        # default_reward is reference so, reset components (multiply by zero) and add hv_reference to get hv_reference.
        hv_reference = default_reward.zero_vector + model_data.get('hv_reference')

        # Prepare Graph Types
        graph_types = set()

        # Update 'states_to_observe' data
        states_to_observe = dict()
        for item in model_data.get('states_to_observe'):

            # Get graph type
            key = GraphType.from_string(item.get('key'))
            graph_types.add(key)

            value = item.get('value')
            key_state = um.lists_to_tuples(value.get('key'))
            value = value.get('value')

            if key not in states_to_observe.keys():
                states_to_observe.update({
                    key: dict()
                })

            states_to_observe.get(key).update({
                key_state: value
            })

        # Unpack 'indexes_counter' data
        indexes_counter = dict()
        for item in model_data.get('indexes_counter'):
            # Convert to tuples for hashing
            key = um.lists_to_tuples(item.get('key'))
            value = item.get('value')

            indexes_counter.update({key: value})

        # Unpack 'v' data
        v = dict()
        for item in model_data.get('v'):
            # Convert to tuples to hash
            key = um.lists_to_tuples(item.get('key'))
            value = item.get('value')
            vector_index = value.get('key')
            value = default_reward.zero_vector + value.get('value')

            vector_index = IndexVector(index=vector_index, vector=value)

            if key not in v.keys():
                v.update({key: [vector_index]})
            else:
                previous_data = v.get(key)
                previous_data.append(vector_index)
                v.update({key: previous_data})

        # Unpack 's' data
        s = dict()
        for item in model_data.get('s'):
            # Convert to tuples to hash
            key = um.lists_to_tuples(item.get('key'))
            value = item.get('value')
            action = value.get('key')
            values = [um.lists_to_tuples(x) for x in value.get('value')]

            if key not in s.keys():
                s.update({
                    key: {
                        action: values
                    }
                })
            else:
                previous_data = s.get(key)
                previous_data.update({action: values})
                s.update({key: previous_data})

        # Unpack 'q' data
        q = dict()
        for item in model_data.get('q'):
            # Convert to tuples to hash
            q_state = um.lists_to_tuples(item.get('key'))
            value = item.get('value')
            q_action = value.get('key')
            value = value.get('value')
            q_index = um.lists_to_tuples(value.get('key'))
            value = value.get('value')

            index_vector = IndexVector(index=value[0], vector=default_reward.zero_vector + value[1])

            if q_state not in q.keys():
                q.update({
                    q_state: dict()
                })

            q_dict_state = q.get(q_state)

            if q_action not in q_dict_state.keys():
                q_dict_state.update({
                    q_action: {
                        q_index: index_vector
                    }
                })
            else:
                q_dict_state.get(q_action).update({
                    q_index: index_vector
                })

        # Prepare an instance of model.
        model = AgentA1(environment=environment, epsilon=epsilon, alpha=alpha, gamma=gamma, seed=seed,
                        max_steps=max_steps, hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism,
                        graph_types=graph_types, integer_mode=integer_mode)

        # Set finals settings and return it.
        model.v = v
        model.s = s
        model.q = q
        model.graph_info = states_to_observe
        model.state = state
        model.total_episodes = total_episodes
        model.total_steps = total_steps

        return model

    def calc_neighbours(self, from_state: tuple, distance: int = 1) -> set:
        """
        Only to mesh environments with actions RIGHT and DOWN (TODO: do generic method)
        :param from_state:
        :param distance:
        :return:
        """
        # Decompose from state
        x_state, y_state = from_state

        current_neighbours = {
            (x_state, y_state - 1),
            # (x_state, y_state + 1),
            (x_state - 1, y_state),
            # (x_state + 1, y_state)
        }

        if distance < 1:
            return set()
        elif distance < 2:
            return current_neighbours
        else:
            all_neighbours = set().union(
                *map(lambda x: self.calc_neighbours(from_state=x, distance=distance - 1), current_neighbours)
            ) - {from_state}

        return all_neighbours
