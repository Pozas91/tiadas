"""
Base Agent class, other agent classes inherited from this.
"""
import json
import os
import time
from pathlib import Path
from pprint import pprint

import matplotlib
import numpy as np

import configurations as conf
import utils.miscellaneous as um
from environments import Environment
from models import GraphType, AgentType


class Agent:
    # Each unit the agent get train_data
    interval_to_get_data = 1
    # Graph types available depending to the agent
    available_graph_types = set()

    def __init__(self, environment: Environment, gamma: float = 1., seed: int = 0, states_to_observe: set = None,
                 max_steps: int = None, graph_types: set = None, initial_value: object = None):
        """
        :param environment: the agent's environment.
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

    def update_graph(self, graph_type: GraphType) -> None:
        """
        Update specific graph type
        :param graph_type:
        :return:
        """
        raise NotImplemented

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

    def reset_graph_info(self):
        """
        Reset states to observe
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

    def show_graph_info(self) -> None:
        """
        Show graph extra
        :return:
        """
        pprint(self.graph_info)

    def print_information(self) -> None:
        """
        Print basic information about agent
        :return:
        """
        print('- Agent information! -')
        print("Seed: {}".format(self.initial_seed))
        print("Gamma: {}".format(self.gamma))

    def time_train(self, execution_time: int, graph_type: GraphType):
        """
        Return this agent trained during `time_execution` seconds.
        :param graph_type:
        :param execution_time:
        :return:
        """

        while (time.time() - self.reference_time_to_train) < execution_time:
            # Do an iteration
            self.do_iteration(graph_type=graph_type)

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
            'train_data': {
                'gamma': self.gamma,
                'environment': {
                    'meta': {
                        'class': self.environment.__class__.__name__,
                        'module': self.environment.__module__,
                    },
                    'train_data': self.environment.get_dict_model()
                },
                'max_steps': self.max_steps,
                'states_to_observe': [
                    {
                        'key': str(k), 'value': {'key': k2 if isinstance(k2, int) else list(k2), 'value': v2}
                    } for k, v in self.graph_info.items() for k2, v2 in v.items()
                ],
                'initial_seed': self.initial_seed
            }
        }

        return model

    def to_json(self) -> str:
        """
        Get a dict model from current object and return as json string.
        :return:
        """
        model = self.get_dict_model()
        return json.dumps(model, indent=conf.json_indent)

    def json_filename(self) -> str:
        """
        Generate a filename for json dump file
        :return:
        """
        # Get environment name in snake case
        env_name_snake = um.str_to_snake_case(self.environment.__class__.__name__)

        # Get only first letter of each word
        env_name_abbr = ''.join([word[0] for word in env_name_snake.split('_')])

        # Get evaluation mechanism in snake case
        agent_name = um.str_to_snake_case(self.__class__.__name__).split('_')[1]

        # Get timestamp
        timestamp = int(time.time())

        return '{}_{}_{}'.format(agent_name, env_name_abbr, timestamp)

    def save(self, filename: str = None) -> None:
        """
        Save model into json file.
        :param filename: If is None, then get current timestamp as filename (defaults 'dumps' dir).
        :return:
        """

        if filename is None:
            filename = self.json_filename()

        # Prepare file path
        file_path = conf.models_path.joinpath(filename)

        # If any parents doesn't exist, make it.
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Get dict model
        model = self.get_dict_model()

        # Open file with filename in write mode with UTF-8 encoding.
        with file_path.open(mode='w', encoding='UTF-8') as file:
            json.dump(model, file, indent=conf.json_indent)

    @staticmethod
    def load(filename: str = None, **kwargs) -> object:
        """
        This method load an agent from filename given.
        :param filename: If is None, then get last timestamp file from 'dumps/models' dir.
        :return:
        """

        if not filename:
            try:
                # Get last file from models directory (higher timestamp)
                filename = sorted(
                    filter(
                        # Get only filenames for this agent (first part of name)
                        lambda path: AgentType.from_string(path.name.split('_')[0]) is kwargs['agent_type'],
                        os.scandir(conf.models_path)
                    ),
                    # Order filenames by timestamp (last part of name)
                    key=lambda path: int(path.name.split('_')[-1])
                )[-1]
            except Exception as error:
                print('Cannot read from {} directory because: {}'.format(conf.models_path, error))

        # Read file from path
        model_path = Path(filename)
        model_file = model_path.open(mode='r', encoding='UTF-8')

        # Load structured train_data from indicated file.
        model = json.load(model_file)

        # Close file
        model_file.close()

        return model

    @staticmethod
    def models_dumps_file_path(filename: str) -> Path:
        # Return path from file name
        return conf.models_path.joinpath(filename)

    def do_iteration(self, graph_type: GraphType) -> None:
        """
        Does an iteration
        :return:
        """
        raise NotImplemented

    def seed(self, seed: int = None) -> None:
        self.generator = np.random.RandomState(seed=seed)
