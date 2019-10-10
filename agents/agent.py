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
from environments import Environment
from models import GraphType


class Agent:
    # Indent of the JSON file where the agent will be saved
    json_indent = 2
    # Get dumps path from this file path
    dumps_path = Path('{}/../../dumps/models'.format(__file__))
    # Each unit the agent get data
    interval_to_get_data = 1

    def __init__(self, environment: Environment, epsilon: float = 0.1, gamma: float = 1., seed: int = 0,
                 states_to_observe: list = None, max_steps: int = None, graph_types: set = None,
                 initial_q_value: object = None):
        """
        :param environment: the agent's environment.
        :param epsilon: Epsilon used in epsilon-greedy policy to control exploration.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param states_to_observe: List of states from which graphical output is provided.
        :param max_steps: Limit of steps per episode.
        """

        # Types to make graphs
        if graph_types is None:
            graph_types = {GraphType.EPISODES, GraphType.STEPS}

        # Discount factor
        assert 0 <= gamma <= 1
        # Exploration factor
        assert 0 <= epsilon <= 1

        self.gamma = gamma
        self.epsilon = epsilon

        # Set environment
        self.environment = environment

        # To intensive problems
        self.max_steps = max_steps

        # Create dictionary with graph information
        graph_info = dict()

        if states_to_observe is not None:
            # Create graph information hierarchy
            graph_info = {
                graph_type: {
                    state: list() for state in states_to_observe
                } for graph_type in graph_types - {GraphType.MEMORY}
            }

        if GraphType.MEMORY in graph_types:
            graph_info.update({
                GraphType.MEMORY: list()
            })

        if GraphType.VECTORS_PER_CELL in graph_types:
            graph_info.update({
                GraphType.VECTORS_PER_CELL: list()
            })

        self.graph_info = graph_info

        # Current Agent State if the initial state of environment
        self.state = self.environment.initial_state

        # Initialize Random Generator with `seed` as initial seed.
        self.seed = seed
        self.generator = np.random.RandomState(seed=seed)

        # Total of this agent
        self.total_steps = 0

        # Steps per episode
        self.steps = 0

        # Initial execution time
        self.last_time_to_get_graph_data = time.time()

        # Total of this agent
        self.total_episodes = 0

        # Initial Q value, when doesn't found another solution.
        self.initial_q_value = initial_q_value

        # Time as reference to mark vectors for graphs
        self.reference_time_to_train = None

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
            action = self._greedy_action(state)

        else:
            # Get best action to exploit reward.
            action = self.best_action(state=state)

        return action

    def _greedy_action(self, state, info=None) -> int:
        """
        Select action according to the greedy policy. The default method is to randomly sample the
        action_space in the environment. The method accepts an optional argument info intended for
        agent dependent information, possibly shared with the method best_action
        :param state:
        :param info: agent dependent information (optional)
        :return:
        """
        return self.environment.action_space.sample()

    def episode(self, graph_type: GraphType) -> None:
        """
        Run an episode complete until get a final step
        :return:
        """

        # Increment total episodes
        self.total_episodes += 1

        # Reset environment
        self.state = self.environment.reset()

        # Condition to stop episode
        is_final_state = False

        # Reset steps
        self.reset_steps()

        while not is_final_state:
            # Do an iteration
            is_final_state = self.do_iteration()

            # Stop conditions
            is_final_state |= (self.max_steps is not None and self.steps >= self.max_steps)

            # Update Graph
            if (graph_type is GraphType.STEPS) and (self.total_steps % self.interval_to_get_data == 0):
                self.update_graph(graph_type=GraphType.STEPS)
            elif (graph_type is GraphType.MEMORY) and (self.total_steps % self.interval_to_get_data == 0):
                self.update_graph(graph_type=GraphType.MEMORY)
            elif graph_type is GraphType.TIME:
                current_time = time.time()

                if (current_time - self.last_time_to_get_graph_data) > self.interval_to_get_data:
                    self.update_graph(graph_type=GraphType.TIME)

                    # Update last execution
                    self.last_time_to_get_graph_data = current_time

    def take_initial_graph_data(self, graph_type: GraphType):
        """
        :param graph_type:
        :return:
        """

        # Check if is necessary update graph
        if graph_type is GraphType.STEPS:
            # Update Graph
            self.update_graph(graph_type=GraphType.STEPS)

        elif graph_type is GraphType.MEMORY:
            # Update Graph
            self.update_graph(graph_type=GraphType.MEMORY)

        elif graph_type is GraphType.EPISODES:
            # Update Graph
            self.update_graph(graph_type=GraphType.EPISODES)

        elif graph_type is GraphType.TIME:
            # Update Graph
            self.update_graph(graph_type=GraphType.TIME)

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

    def best_action(self, state: object = None, info=None) -> int:
        """
        Return best action a state given. The method accepts an optional argument info intended for
        agent dependent information, possibly shared with the method best_action
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
        self.total_episodes = 0

    def reset_graph_info(self):
        """
        Reset states to observe
        :return:
        """
        self.graph_info.update({state: list for state in self.graph_info})

    def show_observed_states(self) -> None:
        """
        Show graph of observed states
        :return:
        """

        for graph_type, states in self.graph_info.items():
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

    def train(self, graph_type: GraphType, limit: int):

        self.reference_time_to_train = time.time()

        # Check if the graph needs to be updated (Before training)
        self.take_initial_graph_data(graph_type=graph_type)

        if graph_type is GraphType.TIME:
            self.time_train(execution_time=limit, graph_type=graph_type)
        elif graph_type is GraphType.EPISODES:
            self.episode_train(episodes=limit, graph_type=graph_type)
        else:
            # In other case, default method is steps training
            self.steps_train(steps=limit, graph_type=graph_type)

        if graph_type is GraphType.VECTORS_PER_CELL:
            # Update Graph
            self.update_graph(graph_type=GraphType.VECTORS_PER_CELL)

    def episode_train(self, episodes: int, graph_type: GraphType):
        """
        Return this agent trained with `episodes` episodes.
        :param graph_type:
        :param episodes:
        :return:
        """

        for i in range(episodes):
            # Do an episode
            self.episode(graph_type=graph_type)

            if (graph_type is GraphType.EPISODES) and (self.total_episodes % self.interval_to_get_data == 0):
                # Update Graph
                self.update_graph(graph_type=GraphType.EPISODES)

    def time_train(self, execution_time: int, graph_type: GraphType):
        """
        Return this agent trained during `time_execution` seconds.
        :param graph_type:
        :param execution_time:
        :return:
        """

        while (time.time() - self.reference_time_to_train) < execution_time:
            # Do an episode
            self.episode(graph_type=graph_type)

    def steps_train(self, steps: int, graph_type: GraphType):
        """
        Return this agent trained during `steps` steps.
        :param graph_type:
        :param steps:
        :return:
        """

        while self.total_steps < steps:
            # Do an episode
            self.episode(graph_type=graph_type)

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
                    } for k, v in self.graph_info.items() for k2, v2 in v.items()
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

        # If any parents doesn't exist, make it.
        file_path.parent.mkdir(parents=True, exist_ok=True)

        # Get dict model
        model = self.get_dict_model()

        # Open file with filename in write mode with UTF-8 encoding.
        with file_path.open('w', encoding='UTF-8') as file:
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
