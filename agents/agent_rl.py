"""
Agent for reinforcement learning agents.
"""
import time

from agents import Agent
from environments import Environment
from models import GraphType


class AgentRL(Agent):
    available_graph_types = {
        GraphType.MEMORY, GraphType.STEPS, GraphType.TIME, GraphType.DATA_PER_STATE, GraphType.EPISODES
    }

    def __init__(self, environment: Environment, epsilon: float = 0.1, gamma: float = 1., seed: int = 0,
                 states_to_observe: list = None, max_steps: int = None, graph_types: set = None,
                 initial_value: object = None):
        """
        :param epsilon: Epsilon used in epsilon-greedy policy to control exploration.
        """

        super().__init__(environment=environment, gamma=gamma, seed=seed, states_to_observe=states_to_observe,
                         max_steps=max_steps, graph_types=graph_types, initial_value=initial_value)

        # Epsilon to exploration
        self.epsilon = epsilon

        # Total of this agent
        self.total_steps = 0

        # Steps per episode
        self.steps = 0

        # Total of this agent
        self.total_episodes = 0

    def select_action(self, state: object = None) -> int:
        """
        Select best action with a little e-greedy policy.
        :return:
        """

        # If position is None, then set current position to position.
        if not state:
            state = self.state

        if self.generator.uniform(low=0., high=1.) < self.epsilon:
            # Get random action to explore possibilities
            action = self._non_greedy_action(state)

        else:
            # Get best action to exploit reward.
            action = self._best_action(state=state)

        return action

    def _non_greedy_action(self, state: object, extra: object = None) -> int:
        """
        Select action according to the greedy policy. The default method is to randomly sample the
        action_space in the environment. The method accepts an optional argument extra intended for
        agent dependent information, possibly shared with the method _best_action
        :param state:
        :param extra: agent dependent information (optional)
        :return:
        """
        return self.environment.action_space.sample()

    def do_iteration(self, graph_type: GraphType) -> None:
        self.episode(graph_type=graph_type)

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
            is_final_state = self.do_step()

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

    def _best_action(self, state: object = None, extra: object = None) -> int:
        """
        Return best action a position given. The method accepts an optional argument extra intended for
        agent dependent information, possibly shared with the method _best_action
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
        self.total_steps = 0
        self.total_episodes = 0

    def print_information(self) -> None:
        super().print_information()
        print("Epsilon: {}".format(self.epsilon))

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

    def get_dict_model(self) -> dict:

        model = super().get_dict_model()

        model['train_data'].update({
            'epsilon': self.epsilon
        })

        return model

    def train(self, graph_type: GraphType, limit: int):

        self.reference_time_to_train = time.time()

        if graph_type is not GraphType.DATA_PER_STATE:
            # Check if the graph needs to be updated (Before training)
            self.update_graph(graph_type=graph_type)

        if graph_type is GraphType.TIME:
            self.time_train(execution_time=limit, graph_type=graph_type)
        elif graph_type is GraphType.EPISODES:
            self.episode_train(episodes=limit, graph_type=graph_type)
        else:
            # In other case, default method is steps training
            self.steps_train(steps=limit, graph_type=graph_type)

        if graph_type is GraphType.DATA_PER_STATE:
            # Update Graph
            self.update_graph(graph_type=GraphType.DATA_PER_STATE)

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

    def do_step(self) -> bool:
        """
        Does a step, and return if the process continues.
        :return:
        """
        raise NotImplemented
