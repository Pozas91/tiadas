"""
Agent W.

The W algorithm is an implementation of an adaptation of the approximation algorithm described by D.J White in
"Multi-objective infinite-horizon discounted Markov decision process".

We consider the case of a finite set of states `S`, a finite set of actions A(state) ∀state ∈ S, and a vector reward `r`
associated to each transition (state, a, state') from position `state` to position `state'` through action `a`.

The algorithm approximates V(state), the set of non-dominated vector values for every position `state`.

We consider the following operations and functions:
• ND(X), the set of non-dominated vectors from vector set X ⊂ R^n.
• r(state, a, state'), the vector reward associated to transition (state, a, state').
• p(state, a, state'), the transition probability associated to transition (state, a, state').
"""
import itertools
import time
from copy import deepcopy
from typing import List

import utils.hypervolume as uh
import utils.miscellaneous as um
import utils.numbers as un
from environments import Environment
from models import Vector, GraphType
from .agent import Agent


class AgentW(Agent):
    available_graph_types = {
        GraphType.MEMORY, GraphType.SWEEP, GraphType.TIME, GraphType.DATA_PER_STATE, GraphType.V_S_0
    }

    def __init__(self, environment: Environment, gamma: float = 1., seed: int = 0, initial_value: Vector = None,
                 states_to_observe: set = None, graph_types: set = None, hv_reference: Vector = None,
                 convergence_graph: bool = False):
        """
        :param environment: An environment where agent does any operation.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param initial_value: Vector with the algorithm begin to learn (by default zero vector).
        :param states_to_observe: List of states from that we want to get a graphical output.
        :param graph_types: Set of types of graph to generate.
        :param hv_reference: Reference vector to calc hypervolume
        :param convergence_graph: If is True then algorithm collects data to draw a convergence graph.
        """

        # Super call __init__
        super().__init__(environment=environment, gamma=gamma, seed=seed, initial_value=initial_value,
                         states_to_observe=states_to_observe, graph_types=graph_types)

        # Check if initial_value is given
        self.initial_q_value = self.environment.default_reward.zero_vector if initial_value is None else initial_value

        # Vector with a vector set for each possible position `state`
        self.v = dict()

        # Total sweeps
        self.total_sweeps = 0

        # Vector reference to calc hypervolume
        self.hv_reference = hv_reference

        # Set if we want the graph of the convergence
        self.convergence_graph = convergence_graph
        self.convergence_graph_data = list()

    def train(self, graph_type: GraphType = None, **kwargs):
        """
        Method to train this agent.
        :param graph_type:
        :param kwargs:
        :return:
        """

        self.reference_time_to_train = time.time()

        # Check if the graph needs to be updated (Before training)
        self.update_graph(graph_type=graph_type)

        # Extract limit information
        limit = kwargs.pop('limit', None)

        if limit is None:
            tolerance = kwargs.pop('tolerance', None)

            if not tolerance:
                raise ValueError('Must indicated tolerance to convergence training.')

            self.convergence_train(tolerance=tolerance, graph_type=graph_type, **kwargs)
        elif graph_type is GraphType.TIME:
            self.time_train(execution_time=limit)
        else:
            # In other case, default method is episode training
            self.sweep_train(sweeps=limit, graph_type=graph_type)

        if graph_type is GraphType.DATA_PER_STATE:
            # Update Graph
            self.update_graph(graph_type=GraphType.DATA_PER_STATE)

    def time_train(self, execution_time: int):
        """
        Return this agent trained during `time_execution` seconds.
        :param execution_time:
        :return:
        """

        while (time.time() - self.reference_time_to_train) < execution_time:
            # Do an iteration
            self.do_iteration()

            # Get time after
            current_time = time.time()

            if (current_time - self.last_time_to_get_graph_data) > self.interval_to_get_data:
                self.update_graph(graph_type=GraphType.TIME)

                # Update last execution
                self.last_time_to_get_graph_data = current_time

    def sweep_train(self, sweeps: int, graph_type: GraphType = None):
        """
        Return this agent trained with `sweeps` sweeps.
        :param sweeps:
        :param graph_type:
        :return:
        """

        for i in range(sweeps):

            # Do an sweep
            self.do_iteration()

            if (graph_type in (GraphType.MEMORY, GraphType.SWEEP, GraphType.V_S_0)) and (
                    self.total_sweeps % self.interval_to_get_data == 0
            ):
                self.update_graph(graph_type=graph_type)

    def convergence_train(self, tolerance: float, graph_type: GraphType = None, **kwargs):
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

            # Do an sweep
            self.do_iteration()

            # Update graph info
            if (graph_type in (GraphType.MEMORY, GraphType.SWEEP, GraphType.V_S_0)) and (
                    self.total_sweeps % self.interval_to_get_data == 0
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

            # Check if make a save
            sweeps_dump = kwargs.get('sweeps_dump')

            if sweeps_dump and self.total_sweeps % sweeps_dump == 0:
                self.save()

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

        if self.convergence_graph:

            # List of differences
            differences = list()

            for key, vectors_v_k_s in v_k.items():
                # Recover vectors from both V'state
                vectors_v_k_1_s = v_k_1[key]

                # If the checks get here, we calculate the hypervolume
                hv_v_k = uh.calc_hypervolume(vectors=vectors_v_k_s, reference=self.environment.hv_reference)
                hv_v_k_1 = uh.calc_hypervolume(vectors=vectors_v_k_1_s, reference=self.environment.hv_reference)

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
                if not (len(vectors_v_k_s) == len(vectors_v_k_1_s)):
                    break

                # If the checks get here, we calculate the hypervolume
                hv_v_k = uh.calc_hypervolume(vectors=vectors_v_k_s, reference=self.environment.hv_reference)
                hv_v_k_1 = uh.calc_hypervolume(vectors=vectors_v_k_1_s, reference=self.environment.hv_reference)

                # Check if absolute difference is lower than tolerance
                converged = abs(hv_v_k_1 - hv_v_k) < tolerance

                # If difference between HV(V_k(state)) and HV(V_k_1(state)) is greater than tolerance, not converged
                if not converged:
                    break

        return converged

    def update_graph(self, graph_type: GraphType) -> None:
        """
        Update specific graph type
        :return:
        """

        # Check for each type of graph
        if graph_type not in self.available_graph_types:
            raise ValueError('Invalid GraphType {} for this agent.'.format(graph_type))

        elif graph_type is GraphType.MEMORY:

            # Count number of vectors in non dominate dictionary
            self.graph_info[graph_type].append(
                sum(len(vectors) for vectors in self.v.values())
            )

        elif graph_type is GraphType.V_S_0:

            self.graph_info[graph_type].append(
                self.v.get(self.environment.initial_state, [self.initial_q_value])
            )

        elif graph_type is GraphType.DATA_PER_STATE:

            # Extract only states with information
            valid_states = self.v.keys()

            # Generate extra
            data = {state: len(self.v[state]) for state in valid_states}

            # Save that information
            self.graph_info[graph_type].append(data)

        else:

            # In the same for loop, is check if this agent has the graph_type indicated (get dictionary default
            # value)
            for state, data in self.graph_info.get(graph_type, {}).items():
                # Extract V(position) (without operations)
                value = self.v.get(state, [self.initial_q_value])

                # Add information to that train_data
                data.append({
                    'train_data': value,
                    'time': time.time() - self.reference_time_to_train,
                    'iterations': self.total_sweeps
                })

                # Update dictionary
                self.graph_info.get(graph_type).update({state: data})

    def do_iteration(self) -> None:
        """
        Does an iteration (In this case a Sweeps)
        :return:
        """

        # Increment total sweeps
        self.total_sweeps += 1

        # Do a copy of v2
        v2 = self.v.copy()

        # Removes all items from the dictionary
        self.v.clear()

        # For each state available
        for s in self.environment.states():

            # A(state) <- Extract all actions available from position `state`
            self.environment.current_state = s

            # Vector of Empty sets
            t = dict()

            # Get all actions available
            actions = self.environment.action_space.copy()

            # For each a in action_space
            for a in actions:

                # Empty set for this a (T(a))
                t_a = set()

                # Get all reachable states for that pair of (state, a)
                s2_set = self.environment.reachable_states(state=s, action=a)

                lv = list()

                for s2 in s2_set:
                    # If this position is unknown return empty set
                    lv.append(v2.get(s2, [Vector(self.initial_q_value)]))

                # Calc cartesian product of each reachable states
                cartesian_product = itertools.product(*lv)

                for product in cartesian_product:

                    summation = self.environment.default_reward.zero_vector

                    for j, s2 in enumerate(s2_set):
                        # Probability to reach that position
                        p = self.environment.transition_probability(state=s, action=a, next_state=s2)

                        # Reward to reach that position
                        r = self.environment.transition_reward(state=s, action=a, next_state=s2)

                        # Get previous value per gamma
                        previous_value = product[j] * self.gamma

                        # Summation
                        summation += (r + previous_value) * p

                    # T(a) <- T(a) U {.....}
                    t_a = t_a.union({summation})

                    t.update({a: t_a})

            # u_t <- U T(a)
            u_t = set.union(*t.values())

            # Remove duplicates and after transform to list
            u_t = set(map(lambda x: un.round_with_precision(x, Vector.decimal_precision), u_t))

            # V(state) <- filter[u_t]
            self.v.update({
                s: self.filter_vectors(vectors=u_t)
            })

    @staticmethod
    def filter_vectors(vectors: set) -> list:
        # ND[vectors]
        return Vector.m3_max(vectors=vectors)

    def recover_policy(self, initial_state: tuple, objective_vector: Vector, **kwargs) -> List[tuple]:
        """
        Returns a policy in temporary order. This way to recover policy allows recover non-stationary policies.
        :param initial_state: State to begin to calculate the policy
        :param objective_vector: Objective vector to search the policy
        :param kwargs: Extra args.
        :return:
        """

        # Do a deepcopy from state and reset it.
        environment_copy = deepcopy(self.environment)
        self.environment.reset()

        # Extract steps limit
        iterations_limit = kwargs.get('iterations_limit', float('inf'))

        # Define initial state
        s = initial_state

        # Final policy
        policy = list()
        simulate = True

        # Set of si (iterations, state, objective vector)
        states_with_objective = {(0, s, objective_vector)}

        while simulate:

            # Extract next relevant state
            iterations, s, objective_vector = states_with_objective.pop()

            # Increment iterations
            iterations += 1

            # Set current state
            self.environment.current_state = s

            # Get all actions available
            actions = self.environment.action_space.copy()

            # Best parameters
            lowest_distance = float('inf')

            # Best options
            best_possible_objective = None

            # Desirable action
            desirable_action = tuple()

            # For a in actions
            for a in actions:
                # Get reachable states
                reachable_states = self.environment.reachable_states(state=s, action=a)

                # Extract v(si) for each si in S
                v_reachable_states = {
                    next_state: self.v.get(next_state, [self.environment.default_reward.zero_vector])
                    for next_state in reachable_states
                }

                # Calc cartesian product
                cartesian_product = itertools.product(*v_reachable_states.values())

                # For each cartesian product
                for product in cartesian_product:

                    # Summation
                    summation = self.environment.default_reward.zero_vector

                    for j, s2 in enumerate(reachable_states):
                        # Probability to reach that position
                        p = self.environment.transition_probability(state=s, action=a, next_state=s2)

                        # Reward to reach that position
                        r = self.environment.transition_reward(state=s, action=a, next_state=s2)

                        # Get previous value per gamma
                        previous_value = product[j] * self.gamma

                        # Summation
                        summation += (r + previous_value) * p

                    # Nearest
                    euclidean_distance = um.euclidean_distance(objective_vector, summation)

                    # Check new information
                    if euclidean_distance <= lowest_distance:
                        # Update lowest distance
                        lowest_distance = euclidean_distance
                        desirable_action = (s, a)

                        # Prepare next states to recover_policy, except if it are final states.
                        best_possible_objective = {
                            (iterations, next_s, product[i]) for i, next_s in enumerate(v_reachable_states)
                            if not self.environment.is_final(state=next_s)
                        }

            # Append desirable action to policy list
            policy.append(desirable_action)

            # Update best possible options
            states_with_objective = states_with_objective.union(best_possible_objective)

            # Filter all states that no exceed limits of iterations
            states_with_objective = set(filter(lambda x: x[0] < iterations_limit, states_with_objective))

            # Continue with the simulation?
            simulate = len(states_with_objective) > 0

        # Restore original environment
        self.environment = environment_copy

        # Return the policy
        return policy
