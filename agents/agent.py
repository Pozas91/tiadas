"""
Base Agent class, other agent classes inherited from this.
"""
import json
import time
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import utils.miscellaneous as um
from gym_tiadas.gym_tiadas.envs import Environment
from models import GraphType


class Agent:
    # Indent of the JSON file where the agent will be saved
    json_indent = 2
    # Get dumps path from this file path
    dumps_path = Path('{}/../../dumps/models'.format(__file__))
    # Each steps to calc graph data
    steps_to_get_graph_data = 1
    # Each seconds to calc graph data
    seconds_to_get_graph_data = 0.001

    def __init__(self, environment: Environment, epsilon: float = 0.1, gamma: float = 1., seed: int = 0,
                 states_to_observe: list = None, max_steps: int = None,
                 graph_types=None):
        """
        :param environment: An environment where agent does any operation.
        :param epsilon: Epsilon using in e-greedy policy, to explore more states.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param states_to_observe: List of states from that we want to get a graphical output.
        :param max_steps: Limits of steps per episode.
        :
        """

        # Types to make graphs
        if graph_types is None:
            graph_types = {GraphType.EPOCHS, GraphType.STEPS}

        # Discount factor
        assert 0 < gamma <= 1
        # Exploration factor
        assert 0 < epsilon <= 1

        self.gamma = gamma
        self.epsilon = epsilon

        # Set environment
        self.environment = environment

        # To intensive problems
        self.max_steps = max_steps

        # Create dictionary of states to observe
        if states_to_observe is None:
            self.states_to_observe = dict()
        else:
            self.states_to_observe = {x: {state: list() for state in states_to_observe} for x in graph_types}

        # Current Agent State if the initial state of environment
        self.state = self.environment.initial_state

        # Initialize Random Generator with `seed` as initial seed.
        self.seed = seed
        self.generator = np.random.RandomState(seed=seed)

        # Total of this agent
        self.total_steps = 0

        # Steps per epoch
        self.steps = 0

        # Initial execution time
        self.last_time_to_get_graph_data = time.time()

        # Total of this agent
        self.total_epochs = 0

    def select_action(self, state: object = None) -> int:
        """
        Select best action with a little e-greedy policy.
        :return:
        """

        # If state is None, then set current state to state.
        if not state:
            state = self.state

        if self.generator.uniform(low=0., high=1.) < self.epsilon:
            # Get random action to explore possibilities
            action = self.environment.action_space.sample()

        else:
            # Get best action to exploit reward.
            action = self.best_action(state=state)

        return action

    def episode(self) -> None:
        """
        Run an episode complete until get a final step
        :return:
        """

        # Increment total epochs
        self.total_epochs += 1

        # Reset environment
        self.state = self.environment.reset()

        # Condition to stop episode
        is_final_state = False

        # Reset steps
        self.reset_steps()

        while not is_final_state:
            is_final_state = self.do_iteration()

            # Check timeout
            if self.max_steps is not None and not is_final_state:
                is_final_state = self.steps >= self.max_steps

            # Check if is necessary update graph
            if GraphType.STEPS in self.states_to_observe and self.total_steps % self.steps_to_get_graph_data == 0:
                # Append new data
                self.update_graph(graph_type=GraphType.STEPS)

            current_time = time.time()

            if GraphType.TIME in self.states_to_observe and (
                    current_time - self.last_time_to_get_graph_data) > self.seconds_to_get_graph_data:
                # Append new data
                self.update_graph(graph_type=GraphType.TIME)

                # Update last execution
                self.last_time_to_get_graph_data = current_time

        if GraphType.EPOCHS in self.states_to_observe:
            # Append new data
            self.update_graph(graph_type=GraphType.EPOCHS)

        # Save last register data
        if GraphType.STEPS in self.states_to_observe:
            # Append new data
            self.update_graph(graph_type=GraphType.STEPS)

        # Save last register data
        if GraphType.TIME in self.states_to_observe:
            # Append new data
            self.update_graph(graph_type=GraphType.TIME)

            # Update last execution
            self.last_time_to_get_graph_data = time.time()

    def update_graph(self, graph_type: GraphType) -> None:
        """
        Update specific graph type
        :param graph_type:
        :return:
        """
        raise NotImplemented

    def reset(self) -> None:
        """
        Reset agent, forgetting previous q-values
        :return:
        """
        raise NotImplemented

    def best_action(self, state: object = None) -> int:
        """
        Return best action a state given.
        :return:
        """
        raise NotImplemented

    def reset_steps(self) -> None:
        """
        Set steps to zero.
        :return:
        """
        self.steps = 0

    def reset_totals(self) -> None:
        """
        Reset totals counters
        :return:
        """
        self.total_steps = 0
        self.total_epochs = 0

    def reset_states_to_observe(self):
        """
        Reset states to observe
        :return:
        """
        self.states_to_observe.update({state: list for state in self.states_to_observe})

    def show_observed_states(self) -> None:
        """
        Show graph of observed states
        :return:
        """

        for graph_type, states in self.states_to_observe.items():
            for state, data in states.items():
                plt.plot(data, label='State: {}'.format(state))

            plt.xlabel(str(graph_type))
            plt.ylabel('data')

            plt.legend(loc='lower center')

            plt.show()

    def print_information(self) -> None:
        """
        Print basic information about agent
        :return:
        """
        print('- Agent information! -')
        print("Seed: {}".format(self.seed))
        print("Gamma: {}".format(self.gamma))
        print("Epsilon: {}".format(self.epsilon))

    def train(self, epochs=1000):
        """
        Return this agent trained with `epochs` epochs.
        :param epochs:
        :return:
        """

        for _ in range(epochs):
            # Do an episode
            self.episode()

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
                'max_steps': self.max_steps,
                'states_to_observe': [
                    {
                        'key': str(k), 'value': {'key': k2 if isinstance(k2, int) else list(k2), 'value': v2}
                    } for k, v in self.states_to_observe.items() for k2, v2 in v.items()
                ],
                'seed': self.seed
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

    def json_filename(self) -> str:
        """
        Generate a filename for json dump file
        :return:
        """
        # Get environment name in snake case
        environment = um.str_to_snake_case(self.environment.__class__.__name__)

        # Get evaluation mechanism in snake case
        agent = um.str_to_snake_case(self.__class__.__name__)

        # Get timestamp
        timestamp = int(time.time())

        return '{}_{}_{}'.format(agent, environment, timestamp)

    def save(self, filename: str = None) -> None:
        """
        Save model into json file.
        :param filename: If is None, then get current timestamp as filename (defaults 'dumps' dir).
        :return:
        """

        if filename is None:
            filename = self.json_filename()

        # Prepare file path
        file_path = self.dumps_file_path(filename=filename)

        # Get dict model
        model = self.get_dict_model()

        # Open file with filename in write mode with UTF-8 encoding.
        with open(str(file_path), 'w', encoding='UTF-8') as file:
            json.dump(model, file, indent=self.json_indent)

    @staticmethod
    def dumps_file_path(filename: str) -> Path:
        # Return path from file name
        return Agent.dumps_path.joinpath(filename)

    def do_iteration(self) -> bool:
        """
        Does an iteration in an episode, and return if the process continues.
        :return:
        """
        raise NotImplemented
