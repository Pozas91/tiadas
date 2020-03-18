"""
Algorithm MPQ
"""
import datetime
import itertools
import math
import time
from typing import Dict, List, Tuple

import gym
import numpy as np

import utils.hypervolume as uh
import utils.miscellaneous as um
from environments import Environment
from models import Vector, IndexVector, VectorDecimal, GraphType, EvaluationMechanism
from .agent_rl import AgentRL


class AgentMPQ(AgentRL):

    def __init__(self, environment: Environment, hv_reference: Vector, alpha: float = 0.1, epsilon: float = 0.1,
                 gamma: float = 1., seed: int = 0, states_to_observe: set = None, max_steps: int = None,
                 evaluation_mechanism: EvaluationMechanism = EvaluationMechanism.HV, graph_types: set = None,
                 initial_value: VectorDecimal = None, convergence_graph: bool = False):
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
        super().__init__(environment=environment, epsilon=epsilon, gamma=gamma, seed=seed, graph_types=graph_types,
                         states_to_observe=states_to_observe, max_steps=max_steps, initial_value=initial_value)

        if initial_value is None:
            self.initial_q_value = VectorDecimal(self.environment.default_reward.zero_vector)

        # Learning factor
        assert 0 < alpha <= 1
        self.alpha = alpha

        # Dictionary that stores all q values. 
        # Key: position; Value: second level dictionary.
        # Second level dictionary: key: action; value: third level dictionary
        # Third level dictionary: key :index vector (element from cartesian product);
        #                        value: q-vector (instance of class IndexVector)
        self.q = dict()

        # States known by each position and action
        self.s = dict()

        # Return non dominate states for a position given
        self.v = dict()

        # Counter to indexes used by each pair (position, action)
        self.indexes_counter = dict()

        # Set of states that need be updated
        self.states_to_update = set()

        # Evaluation mechanism
        if evaluation_mechanism in (EvaluationMechanism.HV, EvaluationMechanism.PO, EvaluationMechanism.C):
            self.evaluation_mechanism = evaluation_mechanism
        else:
            raise ValueError('Evaluation mechanism does not valid.')

        self.hv_reference = hv_reference

        # Set if we want the graph of the convergence
        self.convergence_graph = convergence_graph
        self.convergence_graph_data = list()

    def get_dict_model(self) -> dict:
        """
        Get a dictionary of model
        In JSON serialize only is valid strings as key on dict, so we convert all numeric keys in strings keys.
        :return:
        """

        # Get parent'state model
        model = super().get_dict_model()

        # Own properties
        model.get('train_data').update({
            'indexes_counter': [
                {
                    'key': list(k), 'value': v
                } for k, v in self.indexes_counter.items()
            ]
        })

        model.get('train_data').update({
            'q': [
                dict(key=list(state), value={
                    'key': int(action), 'value': {
                        'key': list(table_index), 'value': (v3.index, v3.vector.tolist())
                    }
                }) for state, v in self.q.items() for action, v2 in v.items() for table_index, v3
                in v2.items()
            ]
        })

        model.get('train_data').update({
            'state': [
                dict(key=list(state), value={
                    'key': int(action), 'value': v2
                }) for state, v in self.s.items() for action, v2 in v.items()
            ]
        })

        model.get('train_data').update({
            'v': [
                dict(key=list(state), value={
                    'key': int(vector_index), 'value': v2.tolist()
                }) for state, v in self.v.items() for vector_index, v2 in v.items()
            ]
        })

        model.get('train_data').update({'alpha': self.alpha})
        model.get('train_data').update({'hv_reference': self.hv_reference.tolist()})
        model.get('train_data').update({'evaluation_mechanism': str(self.evaluation_mechanism)})
        model.get('train_data').update({'total_episodes': self.total_episodes})
        model.get('train_data').update({'total_steps': self.total_steps})
        model.get('train_data').update({'position': list(self.state)})

        return model

    def do_step(self) -> bool:

        # If the position is unknown, register it.
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

        # Convert to decimal vector
        reward = VectorDecimal(reward)

        # Increment steps done
        self.total_steps += 1
        self.steps += 1

        # if self.total_episodes >= 21238 and (self.state == (0, 0)):
        #     print('Q: \n{} \n\nV: \n{}'.format(self.q, self.v))

        # If next_state is a final position and not is register
        if is_final_state:

            # If not is register in V, register it
            if not self.v.get(next_state):
                self.v.update({
                    next_state: {
                        # By default finals states has a zero vector with a zero index
                        0: self.initial_q_value
                    }
                })

        # S(state) -> All known states with its action for the position given.
        pair_action_states_known_by_state = self.s.get(self.state)

        # S(state, a) -> All known states for position and action given.
        states_known_by_state = pair_action_states_known_by_state.get(action, list())

        # I_s_k
        relevant_indexes_of_next_state = self.relevant_indexes_of_state(state=next_state)

        # S_k in S_{n - 1}
        next_state_is_in_states_known = next_state in states_known_by_state

        # Check if sk not in S, and I_s_k is not empty
        if not next_state_is_in_states_known and relevant_indexes_of_next_state:
            # Q_n = N_n(state, a)
            self.new_operation(state=self.state, action=action, reward=reward, next_state=next_state)

        elif next_state_is_in_states_known:
            # Q_n = U_n(state, a)
            self.update_operation(state=self.state, action=action, reward=reward, next_state=next_state)

        # Check if is necessary update V(state) to improve the performance
        self.check_if_need_update_v()

        # Update position
        self.state = next_state

        return is_final_state

    def update_graph(self, graph_type: GraphType) -> None:
        """
        Update specific graph type
        :return:
        """

        if graph_type is GraphType.MEMORY:

            # Count number of vectors in big Q dictionary
            self.graph_info[graph_type].append(
                sum(len(actions.values()) for states in self.q.values() for actions in states.values())
            )

        elif graph_type is GraphType.DATA_PER_STATE:

            # Get positions on axis x and y
            x = self.environment.observation_space.spaces[0].n
            y = self.environment.observation_space.spaces[1].n

            # Extract only states with information
            valid_states = self.q.keys()

            # By default the size of all states is zero
            z = np.zeros([y, x])

            # Calc number of vectors for each position
            for x, y in valid_states:
                z[y, x] = sum(len(actions.values()) for actions in self.q[(x, y)].values())

            # Save that information
            self.graph_info[graph_type].append(z.tolist())

        else:

            # In the same for loop, is check if this agent has the graph_type indicated (get dictionary default
            # value)
            for state, data in self.graph_info.get(graph_type, {}).items():
                # Extract V(state) (without operations)
                value = list(self.v.get(state, {}).values())

                # Set default value
                value = value if value else [self.initial_q_value]

                # Add information to that train_data
                data.append({
                    'train_data': value,
                    'time': time.time() - self.reference_time_to_train,
                    'iterations': self.total_steps
                })

                # Update dictionary
                self.graph_info[graph_type].update({state: data})

    def _best_hypervolume(self, state: object = None) -> float:
        """
        Return best hypervolume for position given.
        :return:
        """

        # Check if a position is given
        state = state if state else self.environment.current_state

        # Get Q-set from position given for each possible action
        v = list(self.v.get(state, {}).values())

        # If v is empty, default is initial_value variable.
        v = v if v else [self.initial_q_value]

        # Getting hypervolume
        hv = uh.calc_hypervolume(vectors=v, reference=self.hv_reference)

        return hv

    def _best_action(self, state: object = None, extra: object = None) -> int:
        """
        Return best action a position given.
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
        else:
            action = self.pareto_evaluation(state=state)

        return action

    def hypervolume_evaluation(self, state: object) -> int:
        """
        Calc the hypervolume for each action in position given. (HV-PQL)
        :param state:
        :return:
        """

        actions = list()
        max_evaluation = float('-inf')

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from position given for each possible action
            q_set = self.q.get(state, dict()).get(a, {
                (0,): IndexVector(index=0, vector=self.initial_q_value)
            })

            # Filter vector from index vectors
            q_set = [q.vector for q in q_set.values()]

            # Calc hypervolume of Q_set, with reference given
            evaluation = uh.calc_hypervolume(vectors=q_set, reference=self.hv_reference)

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
        Calc the cardinality for each action in position given. (C-PQL)
        :param state:
        :return:
        """

        # List of all Qs
        all_q = list()

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from position given for each possible action
            q_set = self.q.get(state, dict()).get(a, {
                (0,): IndexVector(index=a, vector=self.initial_q_value)
            })

            # for each Q in Q_set(state, a)
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
        Calc the pareto for each action in position given. (PO-PQL)
        :param state:
        :return:
        """

        # List of all Qs
        all_q = list()

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from position given for each possible action
            q_set = self.q.get(state, dict()).get(a, {
                (0,): IndexVector(index=a, vector=self.initial_q_value)
            })

            # for each Q in Q_set(state, a)
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

        # Data position Q(state)
        data_state = self.q.get(state)

        # Data action Q(state, a)
        data_action = data_state.get(action, dict())

        # Control update flag
        need_update_v = False

        for p_prime in cartesian_product:

            # p is p_prime less index of next_state (p = p' / sk)
            p = p_prime[:next_state_position] + p_prime[next_state_position + 1:]

            # p'[s_k]
            next_state_index = p_prime[next_state_position]

            # alpha * (reward + gamma * associate_vector)
            next_q = (reward + v.get(next_state_index) * self.gamma) * self.alpha

            # Q_{n - 1}(state, a, p)
            previous_q = data_state.get(action, {}).get(p)

            # If not exists Q_{n - 1}(state, a, p), then create a new IndexVector with the next index available.
            if not previous_q:
                # Q_{n - 1}(state, a, p)
                previous_q = self.default_vector_value(index=index_counter)

                # Update index counter
                index_counter += 1
                self.indexes_counter[state] = index_counter

            # (1 - alpha) * Q_{n - 1}(state, a, p)
            previous_q = previous_q * (1 - self.alpha)

            # Q(state, a)
            q = IndexVector(index=previous_q.index, vector=previous_q.vector + next_q)

            # Q(state, a, p_prime)
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

        # Prepare to add next_state to known states: S(state, a)
        known_states = self.s.get(state).get(action, list())

        # Save index of next_state (sk)
        next_state_p_position = known_states.index(next_state)

        # Calculate the cartesian product
        cartesian_product, relevant_indexes = self.cartesian_product_of_relevant_indexes(states=known_states)

        # Prepare V to do queries
        v = self.v[next_state]

        # Index counter
        index_counter = self.indexes_counter[state]

        # Data position Q(state)
        data_state = self.q[state]

        # Data action position Q(state, a)
        data_action = data_state.get(action, dict())

        # Control update flag
        need_update_v = False

        # Control of positions used updates to remove orphans vectors
        positions_updated = set()

        # For each tuple in cartesian product (p)
        for p in cartesian_product:

            # Position updated
            positions_updated.add(p)

            # p[s_k]
            next_state_reference_vector_index = p[next_state_p_position]

            # associate vector
            associate_vector = v[next_state_reference_vector_index]

            # alpha * (reward + gamma * associate_vector)
            next_q = (reward + associate_vector * self.gamma) * self.alpha

            # Q_{n - 1}(state, a, p)
            previous_q = data_state.get(action, {}).get(p)

            # If not exists Q_{n - 1}(state, a, p), then create a new IndexVector with the next index available.
            if not previous_q:
                # Q_{n - 1}(state, a, p)
                previous_q = self.default_vector_value(index_counter)

                # Update index counter
                index_counter += 1
                self.indexes_counter[state] = index_counter

            # (1 - alpha) * Q_{n - 1}(state, a, p)
            previous_q = previous_q * (1 - self.alpha)

            # Q(state, a)
            q = IndexVector(index=previous_q.index, vector=previous_q.vector + next_q)

            # Q(state, a, p)
            data_action.update({p: q})
            data_state.update({action: data_action})

            # Is necessary update v vector
            need_update_v = True

        # Get positions from table
        all_positions = set(data_action.keys())

        # Deleting difference between all_positions and positions_updated to del zombies vectors
        for position in all_positions - positions_updated:
            # Removing orphan vector
            del data_state[action][position]

        # Check if is necessary update V vectors
        if need_update_v:
            self.states_to_update.add(state)

    def default_vector_value(self, index):
        """
        This method get a default value if a vector doesn't exist. It'state possible change the zero vector for a
        heuristic function.
        :param index:
        :return:
        """
        return IndexVector(index=index, vector=self.initial_q_value)

    def check_if_need_update_v(self) -> None:

        # For each position that need be updated
        for state in self.states_to_update:
            # Get Q(state)
            q_s = self.q[state]

            # List accumulative to save all vectors
            q_list = list()

            # Save previous position
            previous_state = self.environment.current_state

            # Set new position
            self.environment.current_state = state

            # for each action available in position given
            for a in self.environment.action_space:
                # Q(state, a)
                q_list += q_s.get(a, dict()).values()

            # Get all non dominated vectors -> V(state)
            non_dominated_vectors, _ = self.environment.default_reward.m3_max_2_lists_with_buckets(vectors=q_list)

            v = dict()

            # Sort keys of buckets
            for bucket in non_dominated_vectors:
                bucket.sort(key=lambda x: x.index)

                # Set V values (Getting vector with lower index) (simplified)
                first_index_vector = bucket[0]
                v.update({first_index_vector.index: first_index_vector.vector})

            # Update V(state)
            self.v.update({state: v})

            # Restore position
            self.environment.current_state = previous_state

        # All pending states are updated
        self.states_to_update.clear()

    def relevant_indexes_of_state(self, state: object) -> set:
        """
        Return a set of relevant indexes from V(state)
        :param state:
        :return:
        """
        return set(self.v.get(state, {}).keys())

    def cartesian_product_of_relevant_indexes(self, states: list):
        """
        :param states:
        :return:
        """

        # Getting relevant indexes for each position
        indexes = [self.relevant_indexes_of_state(state=iter_state) for iter_state in states]

        # Cartesian product of that indexes
        cartesian_product = itertools.product(*indexes)

        return cartesian_product, indexes

    def objective_training(self, list_of_vectors: list, graph_type: GraphType = None):
        """
        Train until agent V(0, 0) value is close to objective value.
        :param graph_type:
        :param list_of_vectors:
        :return:
        """

        # Calc current hypervolume
        current_hypervolume = self._best_hypervolume(self.environment.initial_state)

        objective_hypervolume = uh.calc_hypervolume(vectors=list_of_vectors, reference=self.hv_reference)

        while not math.isclose(a=current_hypervolume, b=objective_hypervolume, rel_tol=0.02):
            # Do an episode
            self.episode(graph_type=graph_type)

            # Update hypervolume
            current_hypervolume = self._best_hypervolume(self.environment.initial_state)

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

    def convergence_train(self, tolerance: float, graph_type: GraphType = None):
        """
        Return this agent trained until get convergence.
        :param tolerance:
        :param graph_type:
        :return:
        """

        converged = False
        first = True

        if self.convergence_graph:
            self.convergence_graph_data = list()

        while not converged:

            # V_k
            v_k = self.v.copy()

            # Do an iteration
            self.do_iteration()

            # Update graph info
            if (graph_type in (GraphType.MEMORY, GraphType.EPISODES, GraphType.V_S_0)) and (
                    self.total_episodes % self.interval_to_get_data == 0
            ):
                self.update_graph(graph_type=graph_type)

            # MARK: Calc Convergence

            # V_k+1
            v_k_1 = self.v.copy()

            # If is the first iterations doesn't possible model converge, and
            if not first:
                # Checking if model has converged
                converged = self.has_converged(v_k=v_k, v_k_1=v_k_1, tolerance=tolerance)
            else:
                first = False

    def has_converged(self, v_k: dict, v_k_1: dict, tolerance: float) -> bool:
        """
        Check if a policy has converged
        :param v_k:
        :param v_k_1:
        :param tolerance:
        :return:
        """
        # By default
        converged = False

        # If have different length do nothing
        if len(v_k) != len(v_k_1):
            pass
        elif self.convergence_graph:

            # List of differences
            differences = list()

            for key, vectors_v_k_s in v_k.items():
                # Recover vectors from both V'state
                vectors_v_k_1_s = v_k_1[key]

                # If the checks get here, we calculate the hypervolume
                hv_v_k = uh.calc_hypervolume(
                    vectors=[v for v in vectors_v_k_s.values()], reference=self.environment.hv_reference
                )
                hv_v_k_1 = uh.calc_hypervolume(
                    vectors=[v for v in vectors_v_k_1_s.values()], reference=self.environment.hv_reference
                )

                # Check if absolute difference is lower than tolerance
                differences.append(abs(hv_v_k_1 - hv_v_k))

            max_difference = max(differences)
            converged = max_difference < tolerance
            self.convergence_graph_data.append(max_difference)

        else:
            for key, vectors_v_k_s in v_k.items():

                # If all checks are right, convergence will be True, but at the moment...
                converged = False

                # Recover vectors from both V'state
                vectors_v_k_1_s = v_k_1[key]

                # V_k(state) and V_K_1(state) has different lengths
                len_vectors = len(vectors_v_k_s)
                if not (len_vectors == len(vectors_v_k_1_s)):
                    break
                # If the length of vectors is lower than 1, try with the next state
                elif len_vectors < 1:
                    continue

                # If the checks get here, we calculate the hypervolume
                hv_v_k = uh.calc_hypervolume(
                    vectors=[v for v in vectors_v_k_s.values()], reference=self.environment.hv_reference
                )
                hv_v_k_1 = uh.calc_hypervolume(
                    vectors=[v for v in vectors_v_k_1_s.values()], reference=self.environment.hv_reference
                )

                # Check if absolute difference is lower than tolerance
                converged = abs(hv_v_k_1 - hv_v_k) < tolerance

                # If difference between HV(V_k(state)) and HV(V_k_1(state)) is greater than tolerance, not converged
                if not converged:
                    break

        return converged

    # @staticmethod
    # def load(filename: str = None, environment: Environment = None, evaluation_mechanism: EvaluationMechanism = None):
    #     """
    #     Load json string from path and convert to dictionary.
    #     :param evaluation_mechanism: It is an evaluation mechanism that you want load
    #     :param environment: It is an environment that you want load.
    #     :param filename: If is None, then get last timestamp path from 'dumps' dir.
    #     :return:
    #     """
    #
    #     # Check if filename is None
    #     if filename is None:
    #
    #         # Check if environment is also None
    #         if environment is None:
    #             raise ValueError('If you has not indicated a filename, you must indicate a environment.')
    #
    #         # Check if evaluation mechanism is also None
    #         if evaluation_mechanism is None:
    #             raise ValueError('If you has not indicated a filename, you must indicate a evaluation mechanism.')
    #
    #         # Get environment name in snake case
    #         environment = um.str_to_snake_case(environment.__class__.__name__)
    #
    #         # Get evaluation mechanism name in snake case
    #         evaluation_mechanism_value = um.str_to_snake_case(str(evaluation_mechanism.value))
    #
    #         # Filter str
    #         filter_str = '{}_{}'.format(environment, evaluation_mechanism_value)
    #
    #         # Filter files with that environment and evaluation mechanism
    #         files = filter(lambda f: filter_str in f,
    #                        [path.name for path in os.scandir(AgentMPQ.models_dumps_path) if path.is_file()])
    #
    #         # Files to list
    #         files = list(files)
    #
    #         # Sort list of files
    #         files.sort()
    #
    #         # At least must have a path
    #         if files:
    #             # Get last filename
    #             filename = files[-1]
    #
    #     # Prepare path path
    #     file_path = AgentMPQ.models_dumps_file_path(filename)
    #
    #     # Read path from path
    #     try:
    #         file = file_path.open(mode='r', encoding='UTF-8')
    #     except FileNotFoundError:
    #         return None
    #
    #     # Load structured train_data from indicated path.
    #     model = json.load(file)
    #
    #     # Close path
    #     file.close()
    #
    #     # Get meta-train_data
    #     # model_meta = model.get('meta')
    #
    #     # Get train_data
    #     model_data = model.get('train_data')
    #
    #     # ENVIRONMENT
    #     environment = model_data.get('environment')
    #
    #     # Meta
    #     environment_meta = environment.get('meta')
    #     environment_class_name = environment_meta.get('class')
    #     environment_module_name = environment_meta.get('module')
    #     environment_module = importlib.import_module(environment_module_name)
    #     environment_class_ = getattr(environment_module, environment_class_name)
    #
    #     # Data
    #     environment_data = environment.get('train_data')
    #
    #     # Instance
    #     environment = environment_class_()
    #
    #     # Set environment train_data
    #     for key, value in environment_data.items():
    #
    #         if 'position' in key or 'p_stochastic' in key:
    #             # Convert to tuples to hash
    #             value = um.lists_to_tuples(value)
    #
    #         elif 'default_reward' in key:
    #             # If all elements are int, then default_reward is a integer Vector, otherwise float Vector
    #             value = VectorDecimal(value)
    #
    #         vars(environment)[key] = value
    #
    #     # Set initial_seed
    #     environment.seed(seed=environment.initial_seed)
    #
    #     # Get default reward as reference
    #     default_reward = environment.default_reward
    #
    #     # AgentA1
    #
    #     # Meta
    #
    #     # Prepare module and class to make an instance.
    #     # class_name = model_meta.get('class')
    #     # module_name = model_meta.get('module')
    #     # module = importlib.import_module(module_name)
    #     # class_ = getattr(module, class_name)
    #
    #     # Data
    #     epsilon = model_data.get('epsilon')
    #     gamma = model_data.get('gamma')
    #     alpha = model_data.get('alpha')
    #     total_episodes = model_data.get('total_episodes')
    #     total_steps = model_data.get('total_steps')
    #     state = tuple(model_data.get('position'))
    #     max_steps = model_data.get('max_steps')
    #     seed = model_data.get('initial_seed')
    #
    #     # Recover evaluation mechanism from string
    #     evaluation_mechanism = EvaluationMechanism.from_string(
    #         evaluation_mechanism=model_data.get('evaluation_mechanism'))
    #
    #     # default_reward is reference so, reset components (multiply by zero) and add hv_reference to get hv_reference.
    #     hv_reference = default_reward.zero_vector + model_data.get('hv_reference')
    #
    #     # Prepare Graph Types
    #     graph_types = set()
    #
    #     # Update 'states_to_observe' train_data
    #     states_to_observe = dict()
    #     for item in model_data.get('states_to_observe'):
    #
    #         # Get graph type
    #         key = GraphType.from_string(item.get('key'))
    #         graph_types.add(key)
    #
    #         value = item.get('value')
    #         key_state = um.lists_to_tuples(value.get('key'))
    #         value = value.get('value')
    #
    #         if key not in states_to_observe.keys():
    #             states_to_observe.update({
    #                 key: dict()
    #             })
    #
    #         states_to_observe.get(key).update({
    #             key_state: value
    #         })
    #
    #     # Unpack 'indexes_counter' train_data
    #     indexes_counter = dict()
    #     for item in model_data.get('indexes_counter'):
    #         # Convert to tuples for hashing
    #         key = um.lists_to_tuples(item.get('key'))
    #         value = item.get('value')
    #
    #         indexes_counter.update({key: value})
    #
    #     # Unpack 'v' train_data
    #     v = dict()
    #     for item in model_data.get('v'):
    #         # Convert to tuples to hash
    #         key = um.lists_to_tuples(item.get('key'))
    #         value = item.get('value')
    #         vector_index = value.get('key')
    #         value = default_reward.zero_vector + value.get('value')
    #
    #         vector_index = IndexVector(index=vector_index, vector=value)
    #
    #         if key not in v.keys():
    #             v.update({key: [vector_index]})
    #         else:
    #             previous_data = v.get(key)
    #             previous_data.append(vector_index)
    #             v.update({key: previous_data})
    #
    #     # Unpack 'state' train_data
    #     s = dict()
    #     for item in model_data.get('state'):
    #         # Convert to tuples to hash
    #         key = um.lists_to_tuples(item.get('key'))
    #         value = item.get('value')
    #         action = value.get('key')
    #         values = [um.lists_to_tuples(x) for x in value.get('value')]
    #
    #         if key not in s.keys():
    #             s.update({
    #                 key: {
    #                     action: values
    #                 }
    #             })
    #         else:
    #             previous_data = s.get(key)
    #             previous_data.update({action: values})
    #             s.update({key: previous_data})
    #
    #     # Unpack 'q' train_data
    #     q = dict()
    #     for item in model_data.get('q'):
    #         # Convert to tuples to hash
    #         q_state = um.lists_to_tuples(item.get('key'))
    #         value = item.get('value')
    #         q_action = value.get('key')
    #         value = value.get('value')
    #         q_index = um.lists_to_tuples(value.get('key'))
    #         value = value.get('value')
    #
    #         index_vector = IndexVector(index=value[0], vector=default_reward.zero_vector + value[1])
    #
    #         if q_state not in q.keys():
    #             q.update({
    #                 q_state: dict()
    #             })
    #
    #         q_dict_state = q.get(q_state)
    #
    #         if q_action not in q_dict_state.keys():
    #             q_dict_state.update({
    #                 q_action: {
    #                     q_index: index_vector
    #                 }
    #             })
    #         else:
    #             q_dict_state.get(q_action).update({
    #                 q_index: index_vector
    #             })
    #
    #     # Prepare an instance of model.
    #     model = AgentMPQ(environment=environment, epsilon=epsilon, alpha=alpha, gamma=gamma, seed=seed,
    #                      max_steps=max_steps, hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism,
    #                      graph_types=graph_types)
    #
    #     # Set finals settings and return it.
    #     model.v = v
    #     model.s = s
    #     model.q = q
    #     model.graph_info = states_to_observe
    #     model.state = state
    #     model.total_episodes = total_episodes
    #     model.total_steps = total_steps
    #
    #     return model

    def recover_policy(self, **kwargs) -> Tuple[List, Dict]:

        # Extract initial state
        s: tuple = kwargs['initial_state']

        # Extract objective vector
        objective_vector: IndexVector = kwargs['objective_vector']

        # Final trace
        trace = list()
        policy = dict()
        simulate = True

        # Set of si
        states_with_objective = {(s, objective_vector)}

        while simulate:
            # Extract next relevant state
            s, objective_vector = states_with_objective.pop()

            # Search objective index
            selected_action, selected_position = self.search_objective_vector_index(objective_vector.index, s)

            # States trace
            policy.update({s: selected_action})

            # Extract known states data
            for state_tuple_position, known_state in enumerate(self.s[s][selected_action]):
                # Extract the next index to search
                index_to_search = selected_position[state_tuple_position]

                # Check if we know the state indicated (maybe is a final state)
                if self.q.get(known_state, False):
                    # Search objective index
                    selected_action_vector, selected_position_vector = self.search_objective_vector_index(
                        index=index_to_search, state=known_state
                    )

                    next_objective_vector = self.q.get(known_state).get(selected_action_vector).get(
                        selected_position_vector
                    )

                    # Add to next states with objective the vector founded.
                    states_with_objective.add((
                        known_state,
                        next_objective_vector
                    ))

                    trace.append((known_state, index_to_search, next_objective_vector))

            # Check if states_with_objective is not empty
            simulate = len(states_with_objective) > 0

        return trace, policy

    def search_objective_vector_index(self, index: int, state: tuple) -> Tuple[int, tuple]:
        """
        Search the objective vector in the state given
        :param index:
        :param state:
        :return:
        """

        selected_action = None
        selected_position = None

        found = False

        for action in self.q[state]:
            # Extract all vectors with it position
            for position, vector in self.q[state][action].items():
                if vector.index == index:
                    found = True
                    selected_action = action
                    selected_position = position

                if found:
                    break

            if found:
                break

        return selected_action, selected_position

    def evaluate_policy(self, policy: dict, tolerance: float = 0.) -> Dict:

        # Initialize all vectors to zero
        vectors = {s: self.environment.default_reward.zero_vector for s in policy.keys()}

        # Initialize looping variable
        looping = True

        # Continue until A do not change more than tolerance variable
        while looping:
            # A <- 0
            variation = self.environment.default_reward.zero_vector

            # For each state in S
            for s in policy.keys():
                # Initial v
                v = vectors[s]
                a = policy[s]

                # Initialize summation to zero vector
                summation = self.environment.default_reward.zero_vector

                for next_s in self.environment.reachable_states(state=s, action=a):
                    # Calc probability
                    p = self.environment.transition_probability(state=s, action=a, next_state=next_s)
                    # Calc reward
                    r = self.environment.transition_reward(state=s, action=a, next_state=next_s)
                    # v'
                    next_v = vectors.get(next_s, self.environment.default_reward.zero_vector)
                    # Summation
                    summation += ((next_v * self.gamma) + r) * p

                # V(state) <- summation
                vectors[s] = summation

                # A <- max(A, |v - V(state)|)
                abs_difference = abs(v - vectors[s])
                variation = max(variation, abs_difference)

            # Update looping variable
            tolerance_checking = [component < tolerance for component in variation]
            looping = not all(tolerance_checking)

        return vectors
