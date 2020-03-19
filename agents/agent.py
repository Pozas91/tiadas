"""
Base Agent class, other agent classes inherited from this.
"""

import time
from pprint import pprint
from typing import List, Dict

import numpy as np

import utils.miscellaneous as um
import utils.models as u_models
from configurations.paths import dumps_path
from environments import Environment
from models import GraphType, Vector


class Agent:
    # Each unit the agent get train_data
    interval_to_get_data = 1
    # Graph types available depending to the agent
    available_graph_types = set()

    def __init__(self, environment: Environment, gamma: float = 1., seed: int = 0, states_to_observe: set = None,
                 max_steps: int = None, graph_types: set = None, initial_value: object = None):
        """
        :param environment: the agent'state environment.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param states_to_observe: List of states from which graphical output is provided.
        :param max_steps: Limit of steps per episode.
        :param graph_types: Types of graphs where we want extract extra
        :param initial_value: default value for vectors and another calc.
        """

        # Types to make graphs
        if graph_types is None:
            graph_types = set()

        # Set gamma
        self.gamma = gamma

        # Set environment
        self.environment = environment

        # To intensive problems
        self.max_steps = max_steps

        # States to observe
        self.states_to_observe = states_to_observe

        # Graph types
        self.graph_types = graph_types

        # Initialize graph extra
        self.graph_info = dict()
        self.reset_graph_info()

        # Current Agent State if the initial position of environment
        self.state = self.environment.initial_state

        # Initialize Random Generator with `initial_seed` as initial seed.
        self.generator = None
        self.initial_seed = seed
        self.seed(seed=seed)

        # Initial execution time
        self.last_time_to_get_graph_data = time.time()

        # Initial Q value, when doesn't found another solution.
        self.initial_q_value = initial_value

        # Time as reference to mark vectors for graphs
        self.reference_time_to_train = None

    def seed(self, seed: int = None) -> None:
        self.generator = np.random.RandomState(seed=seed)

    # MARK: Information

    def update_graph(self, graph_type: GraphType) -> None:
        """
        Update specific graph type
        :param graph_type:
        :return:
        """
        raise NotImplemented

    def show_graph_info(self) -> None:
        """
        Show for console "raw" graph information
        :return:
        """
        pprint(self.graph_info)

    def print_information(self) -> None:
        """
        Show for console basic information about agent
        :return:
        """
        print('- Agent information! -')
        print("Seed: {}".format(self.initial_seed))
        print("Gamma: {}".format(self.gamma))

    # MARK: Resets

    def reset(self) -> None:
        """
        Reset agent
        :return:
        """
        # Reset initial seed
        self.seed(seed=self.initial_seed)

    def reset_totals(self) -> None:
        """
        Reset totals counters
        :return:
        """
        raise NotImplemented

    def reset_graph_info(self) -> None:
        """
        Reset the structure where store graph information
        :return:
        """

        if self.states_to_observe is not None:
            # Create graph information hierarchy
            self.graph_info = {
                graph_type: {
                    state: list() for state in self.states_to_observe
                } for graph_type in self.graph_types - {GraphType.MEMORY}
            }

        if GraphType.MEMORY in self.graph_types:
            self.graph_info.update({
                GraphType.MEMORY: list()
            })

        if GraphType.V_S_0 in self.graph_types:
            self.graph_info.update({
                GraphType.V_S_0: list()
            })

        if GraphType.DATA_PER_STATE in self.graph_types:
            self.graph_info.update({
                GraphType.DATA_PER_STATE: list()
            })

    # MARK: Train model

    def do_iteration(self, **kwargs) -> None:
        """
        Does an iteration
        :return:
        """
        raise NotImplemented

    def train(self, **kwargs) -> None:
        """
        Train this model
        """
        raise NotImplemented

    # MARK: Dumps model

    def default_filename(self, mode: str = 'binary') -> str:
        """
        Return default filename
        :return:
        """

        # Get environment name in snake case, get only first letter of each word
        env_str_abbr = ''.join(
            word[0] for word in
            um.str_to_snake_case(self.environment.__class__.__name__).split('_')
        )

        # Extract agent name
        agent_str_abbr = um.str_to_snake_case(self.__class__.__name__).split('_')[-1]

        # Get timestamp
        timestamp = int(time.time())

        if mode == 'binary':
            extension = 'bin'
        else:
            extension = ''

        # Prepare default filename
        return '{}/models/{}_{}_{}.{}'.format(
            agent_str_abbr, env_str_abbr, timestamp, Vector.decimal_precision, extension
        )

    @staticmethod
    def load(filename: str, mode: str = 'binary', **kwargs) -> 'Agent':
        """
        This method load an agent from filename given.
        At moment only can load in binary mode
        :param filename: Filename from load the model.
        :param mode:
        :return:
        """

        if mode == 'binary':
            agent = u_models.load(path=dumps_path.joinpath(filename))
        else:
            raise ValueError('Indicate mode don\'t recognize.')

        return agent

    def save(self, mode: str = 'binary', filename: str = None) -> None:
        """
        Dumps model given into dumps directory
        At moment only can save in binary mode
        :param filename:
        :param mode:
        :return:
        """

        if filename is None:
            filename = self.default_filename(mode=mode)

        # Define file path
        file_path = dumps_path.joinpath(filename)

        # If does not exists make it.
        file_path.parent.mkdir(parents=True, exist_ok=True)

        if mode == 'binary':
            # Dumps model in binary mode
            u_models.binary_dump(path=file_path, model=self)
        else:
            raise ValueError('Indicate mode don\'t recognize.')

    def recover_policy(self, initial_state: tuple, objective_vector: Vector, **kwargs) -> List[tuple]:
        """
        Simulate a walking of the agent, and return a dictionary with each state related with an action.
        :return:
        """
        raise NotImplemented

    def evaluate_policy(self, policy: list, tolerance: float = 0.1) -> dict:
        """
        Evaluates a policy given. This method cannot evaluate non-stationary policies.
        :param policy: The policy to evaluate
        :param tolerance: Tolerance to stops of evaluate
        :return:
        """

        # Convert to stationary policy
        policy = {s: a for s, a in policy}

        # Initialize all vectors to zero
        vectors = {s: self.environment.default_reward.zero_vector for s in policy.keys()}

        # Initialize looping variable
        looping = True

        # Continue until A do not change more than tolerance variable
        while looping:

            # For each state in S
            for s, v in vectors.items():
                # Initial v
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

                # Update looping variable
                looping = not all(component < tolerance for component in abs_difference)

                if looping:
                    break

        return vectors
