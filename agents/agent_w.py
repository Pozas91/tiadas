"""
Agent W.

The W algorithm is an implementation of an adaptation of the approximation algorithm described by D.J White in
"Multi-objective infinite-horizon discounted Markov decision process".

We consider the case of a finite set of states `S`, a finite set of actions A(s) ∀s ∈ S, and a vector reward `r`
associated to each transition (s, a, s') from position `s` to position `s'` through action `a`.

The algorithm approximates V(s), the set of non-dominated vector values for every position `s`.

We consider the following operations and functions:
• ND(X), the set of non-dominated vectors from vector set X ⊂ R^n.
• r(s, a, s'), the vector reward associated to transition (s, a, s').
• p(s, a, s'), the transition probability associated to transition (s, a, s').
"""
import itertools
import time

import utils.hypervolume as uh
import utils.numbers as un
from environments import Environment
from models import Vector, GraphType, AgentType
from .agent import Agent


class AgentW(Agent):
    available_graph_types = {
        GraphType.MEMORY, GraphType.SWEEP, GraphType.TIME, GraphType.DATA_PER_STATE, GraphType.V_S_0
    }

    def __init__(self, environment: Environment, gamma: float = 1., seed: int = 0, initial_value: Vector = None,
                 states_to_observe: set = None, graph_types: set = None, hv_reference: Vector = None,
                 convergence_graph: bool = False):

        # Super call __init__
        super().__init__(environment=environment, gamma=gamma, seed=seed, initial_value=initial_value,
                         states_to_observe=states_to_observe, graph_types=graph_types)

        # Check if initial_value is given
        self.initial_q_value = self.environment.default_reward.zero_vector if initial_value is None else initial_value

        # Vector with a vector set for each possible position `s`
        self.v = dict()

        # Total sweeps
        self.total_sweeps = 0

        # Vector reference to calc hypervolume
        self.hv_reference = hv_reference

        # Set if we want the graph of the convergence
        self.convergence_graph = convergence_graph
        self.convergence_graph_data = list()

    def train(self, graph_type: GraphType = None, **kwargs):

        self.reference_time_to_train = time.time()

        # Check if the graph needs to be updated (Before training)
        self.update_graph(graph_type=graph_type)

        # Extract limit information
        limit = kwargs.get('limit', None)

        if limit is None:
            tolerance = kwargs.get('tolerance', None)

            if not tolerance:
                raise ValueError('Must indicated tolerance to convergence training.')

            self.convergence_train(tolerance=tolerance, graph_type=graph_type)
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

    def has_converged(self, v_k: dict, v_k_1: dict, tolerance: float) -> bool:

        # By default
        converged = False

        if self.convergence_graph:

            # List of differences
            differences = list()

            for key, vectors_v_k_s in v_k.items():
                # Recover vectors from both V's
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

                # Recover vectors from both V's
                vectors_v_k_1_s = v_k_1[key]

                # V_k(s) and V_K_1(s) has different lengths
                if not (len(vectors_v_k_s) == len(vectors_v_k_1_s)):
                    break

                # If the checks get here, we calculate the hypervolume
                hv_v_k = uh.calc_hypervolume(vectors=vectors_v_k_s, reference=self.environment.hv_reference)
                hv_v_k_1 = uh.calc_hypervolume(vectors=vectors_v_k_1_s, reference=self.environment.hv_reference)

                # Check if absolute difference is lower than tolerance
                converged = abs(hv_v_k_1 - hv_v_k) < tolerance

                # If difference between HV(V_k(s)) and HV(V_k_1(s)) is greater than tolerance, not converged
                if not converged:
                    break

        return converged

    def update_graph(self, graph_type: GraphType) -> None:

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

        # Increment total sweeps
        self.total_sweeps += 1

        # Do a copy of v2
        v2 = self.v.copy()

        # Removes all items from the dictionary
        self.v.clear()

        # For each state available
        for s in self.environment.states():

            # A(s) <- Extract all actions available from position `s`
            self.environment.current_state = s

            # Vector of Empty sets
            t = dict()

            # Get all actions available
            actions = self.environment.action_space.copy()

            # For each a in action_space
            for a in actions:

                # Empty set for this a (T(a))
                t_a = set()

                # Get all reachable states for that pair of (s, a)
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

            # V(s) <- ND[U T(a)]
            u_t = set.union(*t.values())
            u_t = list(map(lambda x: un.round_with_precision(x, Vector.decimal_precision), u_t))
            self.v.update({s: Vector.m3_max(u_t)})

    @staticmethod
    def load(filename: str = None, **kwargs) -> object:
        # TODO: Finish this method
        return Agent.load(filename=filename, agent_type=AgentType.W)
