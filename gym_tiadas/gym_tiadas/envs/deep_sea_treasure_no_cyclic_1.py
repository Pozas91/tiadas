"""The environment is a grid of 10 rows and 11 columns. The agent controls a submarine searching for undersea
treasure. There are multiple treasure locations with varying values. There are two objectives - to minimise the time
taken to reach the treasure, and to maximise the value of the treasure. Each episode commences with the vessel in the
top left state, and ends when a treasure location is reached or after 1000 actions. Four actions are available to the
agent - moving one square to the left, right, up or down. Any action which would cause the agent to leave the grid
will leave its position unchanged. The reward received by the agent is a 2-element vector. The first element is a
time penalty, which is -1 on all turns. The second element is the treasure value which is 0 except when the agent
moves into a treasure location, when it is the value indicated.

FINAL STATE: To reach any final state.

REF: Empirical Evaluation methods for multi-objective reinforcement learning algorithms
    (Vamplew, Dazeley, Berry, Issabekov and Dekker) 2011
"""
from models import Vector
from spaces import DynamicSpace
from .env_mesh import EnvMesh


class DeepSeaTreasureNoCyclic1(EnvMesh):
    # Possible actions
    _actions = {'RIGHT': 0, 'DOWN': 1}

    # Pareto optimal
    pareto_optimal = [
        (-2, 5)
    ]

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0,), seed: int = 0):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        """

        # List of all treasures and its reward.
        finals = {
            (1, 1): 5,
        }

        obstacles = frozenset()
        obstacles = obstacles.union([(0, 1)])

        mesh_shape = (2, 2)

        # Default reward plus time (time_inverted, treasure_value)
        default_reward = (-1,) + default_reward
        default_reward = Vector(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, initial_state=initial_state, default_reward=default_reward,
                         finals=finals, obstacles=obstacles)

        # Trying improve performance
        self.dynamic_action_space = DynamicSpace([])
        self.dynamic_action_space.seed(seed=seed)

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (time_inverted, treasure_value), final, info)
        """

        # Initialize rewards as vector
        rewards = self.default_reward.copy()

        # Get new state
        new_state = self.next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Get treasure value
        rewards[1] = self.finals.get(self.current_state, self.default_reward[1])

        # Set info
        info = {}

        # Check is_final
        final = self.is_final(self.current_state)

        return self.current_state, rewards, final, info

    def reset(self) -> tuple:
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        return self.current_state

    def is_final(self, state: tuple = None) -> bool:
        """
        Return True if state given is terminal, False in otherwise.
        :param state:
        :return:
        """
        return state in self.finals.keys()

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next state with current state and action given. Default is 4-neighbors (UP, LEFT, DOWN, RIGHT)
        :param state: If a state is given, do action from that state.
        :param action: from action_space
        :return: a new state (or old if is invalid action)
        """

        # Get my position
        x, y = state if state else self.current_state

        # Do movement
        if action == self._actions.get('RIGHT'):
            x += 1
        elif action == self._actions.get('DOWN'):
            y += 1

        # Set new state
        new_state = x, y

        # If exists obstacles, then new_state must be in self.obstacles
        is_obstacle = bool(self.obstacles) and new_state in self.obstacles

        if not self.observation_space.contains(new_state) or is_obstacle or state == new_state:
            raise ValueError("Action/State combination isn't valid.")

        # Return (x, y) position
        return new_state

    @property
    def action_space(self) -> DynamicSpace:
        """
        Get a dynamic action space with only valid actions.
        :return:
        """

        # Get current state
        x, y = self.current_state

        # Setting possible actions
        possible_actions = []

        # Get all actions available

        # Can we go to right?
        x_right = x + 1

        # Check that x_right is not an obstacle and is into mesh
        if (x_right, y) not in self.obstacles and self.observation_space.contains((x_right, y)):
            # We can go to right
            possible_actions.append(self._actions.get('RIGHT'))

        # Can we go to down?
        y_down = y + 1

        # Check that y_down is not an obstacle and is into mesh
        if (x, y_down) not in self.obstacles and self.observation_space.contains((x, y_down)):
            # We can go to down
            possible_actions.append(self._actions.get('DOWN'))

        # Setting to dynamic_space
        self.dynamic_action_space.items = possible_actions

        # Update n length
        self.dynamic_action_space.n = len(possible_actions)

        # Return a list of iterable valid actions
        return self.dynamic_action_space
