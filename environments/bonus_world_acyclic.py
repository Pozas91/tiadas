# coding=utf-8
"""
Variant of BonusWorld environment to acyclic agents. If agent there on pit state, episode ends and agent receives
(-50, -50) reward.

HV REFERENCE: (-50, -50, -50)
"""
from models import Vector
from spaces import DynamicSpace
from .env_mesh import EnvMesh


class BonusWorldAcyclic(EnvMesh):
    # Possible actions
    _actions = {'RIGHT': 0, 'DOWN': 1}

    # Experiments common hypervolume reference
    hv_reference = Vector([0, 0, -150])

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0, 0), seed: int = 0):
        """
        :param initial_state:
        :param default_reward: (objective 1, objective 2)
        :param seed:
        """

        # List of all treasures and its reward.
        finals = {
            (8, 0): Vector([1, 9]),
            (8, 2): Vector([3, 9]),
            (8, 4): Vector([5, 9]),
            (8, 6): Vector([7, 9]),
            (8, 8): Vector([9, 9]),

            (0, 8): Vector([9, 1]),
            (2, 8): Vector([9, 3]),
            (4, 8): Vector([9, 5]),
            (6, 8): Vector([9, 7]),
        }

        # Define mesh shape
        mesh_shape = (9, 9)

        # Set obstacles
        # obstacles = frozenset([(2, 2), (2, 3), (3, 2)])
        obstacles = frozenset([
            (2, 2)
        ])

        # Default reward plus time (objective 1, objective 2, time)
        default_reward += (-1,)
        default_reward = Vector(default_reward)

        # Separate position from bonus_activated
        initial_state, self.bonus_activated = initial_state

        super().__init__(mesh_shape=mesh_shape, seed=seed, default_reward=default_reward, initial_state=initial_state,
                         finals=finals, obstacles=obstacles)

        # Pits marks which returns the agent to the start location.
        self.pits = [
            (7, 1), (7, 3), (7, 5), (1, 7), (3, 7), (5, 7)
        ]

        # X2 bonus
        self.bonus = [
            (3, 3)
        ]

        # Pits penalize
        self.pits_penalize = Vector([-50, -50, -1])

        # Trying improve performance
        self.dynamic_action_space = DynamicSpace([])
        self.dynamic_action_space.seed(seed=seed)

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (objective 1, objective 2, time), final, info)
        """

        # Initialize rewards as vector
        rewards = self.default_reward.copy()

        # Get new state
        new_state = self.next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Check if the agent has activated the bonus
        if self.current_state in self.bonus:
            self.bonus_activated = True

        # Get treasure value
        rewards[0], rewards[1] = self.finals.get(self.current_state, (self.default_reward[0], self.default_reward[1]))

        # If the bonus is activated, double the reward.
        if self.bonus_activated:
            rewards[0] *= 2
            rewards[1] *= 2

        # If agent is in pit, it's returned at initial state.
        if self.current_state in self.pits:
            self.current_state = self.initial_state
            rewards = self.pits_penalize

        # Set info
        info = {}

        # Check is_final
        final = self.is_final(self.current_state)

        return (self.current_state, self.bonus_activated), rewards, final, info

    def reset(self) -> tuple:
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        self.bonus_activated = False

        # Return ((x, y), bonus)
        return self.current_state, self.bonus_activated

    def is_final(self, state: tuple = None) -> bool:
        """
        Is final if agent is on final state.
        :param state:
        :return:
        """
        return state in self.finals.keys() or state in self.pits

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next state with current state and action given. Default is 2-neighbors (DOWN, RIGHT)
        :param state: If a state is given, do action from that state.
        :param action: from action_space
        :return: a new state (or old if is invalid action)
        """

        # Get my position
        x, y = state if state else self.current_state

        # Do movement
        if action == self._actions['RIGHT']:
            x += 1
        elif action == self._actions['DOWN']:
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

        # Can we go to right?
        x_right = x + 1

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x_right, y)) and (x_right, y) not in self.obstacles:
            # We can go to right
            possible_actions.append(self._actions['RIGHT'])

        # Can we go to down?
        y_down = y + 1

        # Check that y_down is not and obstacle and is into mesh
        if self.observation_space.contains((x, y_down)) and (x, y_down) not in self.obstacles:
            # We can go to down
            possible_actions.append(self._actions['DOWN'])

        # Setting to dynamic_space
        self.dynamic_action_space.items = possible_actions

        # Update n length
        self.dynamic_action_space.n = len(possible_actions)

        # Return a list of iterable valid actions
        return self.dynamic_action_space

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment data
        del data['pits']
        del data['bonus']
        del data['dynamic_action_space']

        return data
