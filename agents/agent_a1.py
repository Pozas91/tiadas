import itertools
import math
from copy import deepcopy

import utils.hypervolume as uh
from agents import Agent
from gym_tiadas.gym_tiadas.envs import Environment
from models import Vector, IndexVector, VectorFloat


class AgentA1(Agent):

    def __init__(self, environment: Environment, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 1.,
                 seed: int = 0, states_to_observe: list = None, max_iterations: int = None,
                 evaluation_mechanism: str = 'HV-PQL', hv_reference: Vector = None):
        """
        :param environment: An environment where agent does any operation.
        :param alpha: Learning rate
        :param epsilon: Epsilon using in e-greedy policy, to explore more states.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param states_to_observe: List of states from that we want to get a graphical output.
        :param max_iterations: Limits of iterations per episode.
        :param hv_reference: Reference vector to calc hypervolume
        :param evaluation_mechanism: Evaluation mechanism used to calc best action to choose. Three values are
            available: 'C-PQL', 'PO-PQL', 'HV-PQL'
        """

        # Super call __init__
        super().__init__(environment=environment, epsilon=epsilon, gamma=gamma, seed=seed,
                         states_to_observe=states_to_observe, max_iterations=max_iterations)

        # Learning factor
        assert 0 < alpha <= 1
        self.alpha = alpha

        # Initialize to Q-Learning values
        self.q = dict()

        # States known by each state and action
        self.s = dict()

        # Return non dominate states for a state given
        self.v = dict()

        # Counter to indexes used by each pair (state, action)
        self.indexes_counter = dict()

        # HV reference
        self.hv_reference = hv_reference * 100

        # Evaluation mechanism
        if evaluation_mechanism in ('HV-PQL', 'PO-PQL', 'C-PQL'):
            self.evaluation_mechanism = evaluation_mechanism
        else:
            raise ValueError('Evaluation mechanism does not valid.')

    def episode(self) -> None:
        """
        Run an episode complete until get a final step
        :return:
        """

        # Reset environment
        self.state = self.environment.reset()

        # Condition to stop episode
        is_final_state = False

        # Reset iterations
        self.reset_iterations()

        while not is_final_state:

            # Increment iterations
            self.iterations += 1

            # TODO: Optimize this part of initialization

            # If the state is unknown, register it.
            if self.state not in self.q:
                self.q.update({self.state: {'updated': True}})

            if self.state not in self.s:
                self.s.update({self.state: dict()})

            if self.state not in self.v:
                self.v.update({self.state: dict()})

            if self.state not in self.indexes_counter:
                # Initialize counters
                self.indexes_counter.update({self.state: 0})

            # END TODO

            # Check if is necessary update V(s) to improve the performance
            self.check_if_need_update_v()

            # Get an action
            action = self.select_action()

            # Do step on environment
            next_state, reward, is_final_state, info = self.environment.step(action=action)

            # Transform to operate with int vectors getting some decimals
            reward *= Vector.decimals

            # If next_state is a final state and not is register
            if is_final_state:

                # If not is register in V, register it
                if not self.v.get(next_state):
                    self.v.update({next_state: {
                        # By default finals states has a zero vector with a zero index
                        0: self.environment.default_reward.zero_vector
                    }})

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

            # Update state
            self.state = next_state

            # Check timeout
            if self.max_iterations is not None and not is_final_state:
                is_final_state = self.iterations >= self.max_iterations

    def best_action(self, state: object = None) -> int:
        """
        Return best action a state given.
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
        cartesian_product, indexes = self.calc_cartesian_product(states=states)

        # Prepare V to do queries
        v = self.v.get(next_state)

        # Index counter
        index_counter = self.indexes_counter.get(state)

        # Data state Q(s)
        data_state = self.q.get(state)

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
                previous_q = IndexVector(index=index_counter, vector=self.environment.default_reward.zero_vector)

                # Update index counter
                index_counter += 1
                self.indexes_counter[state] = index_counter

            # (1 - alpha) * Q_{n - 1}(s, a, p)
            previous_q = previous_q * (1 - self.alpha)

            # Q(s, a)
            q = IndexVector(index=previous_q.index, vector=previous_q.vector + next_q)

            # Q(s, a, p_prime)
            data_action = data_state.get(action, dict())
            data_action.update({p_prime: q})

            # Need delete previous index of table if is necessary
            if p != p_prime:
                data_action.pop(p, None)

            data_state.update({action: data_action})
            data_state.update({'updated': False})

    def update_operation(self, state: object, action: int, reward: Vector, next_state: object) -> None:
        """
        :param state:
        :param action:
        :param reward:
        :param next_state:
        :return:
        """

        # Prepare to add next_state to known states
        states = self.s.get(state).get(action, list())

        # Save index of next_state (sk)
        next_state_position = states.index(next_state)

        # Calculate the cartesian product
        cartesian_product, indexes = self.calc_cartesian_product(states=states)

        # Prepare V to do queries
        v = self.v.get(next_state)

        # Index counter
        index_counter = self.indexes_counter.get(state)

        # Data state Q(s)
        data_state = self.q.get(state)

        for p in cartesian_product:

            # p[s_k]
            next_state_index = p[next_state_position]

            # alpha * (reward + gamma * associate_vector)
            next_q = (reward + v.get(next_state_index) * self.gamma) * self.alpha

            # Q_{n - 1}(s, a, p)
            previous_q = data_state.get(action, {}).get(p)

            # If not exists Q_{n - 1}(s, a, p), then create a new IndexVector with the next index available.
            if not previous_q:
                # Q_{n - 1}(s, a, p)
                previous_q = IndexVector(index=index_counter, vector=self.environment.default_reward.zero_vector)

                # Update index counter
                index_counter += 1
                self.indexes_counter[state] = index_counter

            # (1 - alpha) * Q_{n - 1}(s, a, p)
            previous_q = previous_q * (1 - self.alpha)

            # Q(s, a)
            q = IndexVector(index=previous_q.index, vector=previous_q.vector + next_q)

            # Q(s, a, p)
            data_action = data_state.get(action, dict())
            data_action.update({p: q})
            data_state.update({action: data_action})
            data_state.update({'updated': False})

    def check_if_need_update_v(self) -> None:

        # Get all possible states
        states = self.q.keys()

        # For each state in states
        for state in states:

            # Get Q(s)
            q_s = self.q.get(state)

            # Check if this action need
            updated = q_s.get('updated', True)

            # If V(s) is not updated
            if not updated:

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
                v = self.environment.default_reward.m3_max(vectors=q_list)

                # Update V(s)
                self.v.update({state: {index_vector.index: index_vector.vector for index_vector in v}})

                # V(s) already updated
                self.q.get(state).update({'updated': True})

                # Restore state
                self.environment.current_state = previous_state

    def relevant_indexes_of_state(self, state: object) -> set:
        """
        Return a set of relevant indexes from V(s)
        :param state:
        :return:
        """
        return set(self.v.get(state, {}).keys())

    def calc_cartesian_product(self, states: list):
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

            # Remove updated flag
            action_dict.pop('updated', None)

            # For each action with it indexes dictionary
            for action, indexes_dict in action_dict.items():
                # For each index with it index_vector (with a int vector associated multiplied by Vector.decimals)
                for index, index_vector in indexes_dict.items():
                    # Divide that vector by Vector.decimals to convert in original float vector
                    index_vector.vector = VectorFloat(
                        index_vector.vector.components / self.environment.default_reward.decimals
                    )

                    # Update Q-table dictionary
                    indexes_dict.update({index: index_vector})
                    action_dict.update({action: indexes_dict})
                    q.update({state: action_dict})

        # Return Q-table transformed
        return q

    def v_real(self):
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
                vectors.update({index: VectorFloat(
                    vector.components / self.environment.default_reward.decimals
                )})
                # Update V-values dictionary
                v.update({state: vectors})

        return v