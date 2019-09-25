"""
Inspired by the Deep Sea Treasure (DST) environment. In contrast to the, the values of the treasures are altered to
create a convex Pareto front.

FINAL STATE: To reach any final state.

REF: Multi-objective reinforcement learning using sets of pareto dominating policies (Kristof Van Moffaert,
Ann Nowé) 2014

HV REFERENCE: (-25, 0, -120)
"""
import numpy as np

from models import Vector
from spaces import DynamicSpace
from .env_mesh import EnvMesh
import math


class PressurizedBountifulSeaTreasureRightDownStochastic(EnvMesh):
    # Possible actions
    _actions = {'RIGHT_PROB': 0, 'DOWN_PROB': 1, 'DOWN': 2}

    # Pareto optimal
    pareto_optimal = [
        (-1, 5, -2), (-3, 80, -3), (-5, 120, -4), (-7, 140, -5), (-8, 145, -6), (-9, 150, -6), (-13, 163, -8),
        (-14, 166, -8), (-17, 173, -10), (-19, 175, -11)
    ]

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0,), seed: int = 0, columns: int = 0,
                 transitions: tuple = (0.8, 0.2)):
        """
        :param initial_state:
        :param default_reward: (treasure_value)
        :param seed:
        :param columns:
        :param transitions:
        """

        original_mesh_shape = (10, 11)

        # Reduce the number of columns
        if columns < 1 or columns > original_mesh_shape[0]:
            columns = original_mesh_shape[0]

        # List of all treasures and its reward.
        finals = {
            (0, 1): 5,
            (1, 2): 80,
            (2, 3): 120,
            (3, 4): 140,
            (4, 4): 145,
            (5, 4): 150,
            (6, 7): 163,
            (7, 7): 166,
            (8, 9): 173,
            (9, 10): 175,
        }

        # Filter finals states
        finals = dict(filter(lambda x: x[0][0] < columns, finals.items()))

        obstacles = frozenset()
        obstacles = obstacles.union([(0, y) for y in range(2, 11)])
        obstacles = obstacles.union([(1, y) for y in range(3, 11)])
        obstacles = obstacles.union([(2, y) for y in range(4, 11)])
        obstacles = obstacles.union([(3, y) for y in range(5, 11)])
        obstacles = obstacles.union([(4, y) for y in range(5, 11)])
        obstacles = obstacles.union([(5, y) for y in range(5, 11)])
        obstacles = obstacles.union([(6, y) for y in range(8, 11)])
        obstacles = obstacles.union([(7, y) for y in range(8, 11)])
        obstacles = obstacles.union([(8, y) for y in range(10, 11)])

        # Filter obstacles states
        obstacles = frozenset(filter(lambda x: x[0] < columns, obstacles))

        # Resize mesh_shape
        mesh_shape = (columns, 11)

        # Default reward plus time (time_inverted, treasure_value, water_pressure)
        default_reward = (-1,) + default_reward + (0,)
        default_reward = Vector(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, default_reward=default_reward,
                         initial_state=initial_state, finals=finals, obstacles=obstacles)

        # Trying improve performance
        self.dynamic_action_space = DynamicSpace([])
        self.dynamic_action_space.seed(seed=seed)

        # Prepare stochastic transitions
        assert isinstance(transitions, tuple) and math.isclose(a=np.sum(transitions), b=1) and len(transitions) == 2

        self.transitions = transitions

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

        # Water pressure (y-coordinate)
        rewards[2] = -(self.current_state[1] + 1)

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
        If agent is in treasure
        :param state:
        :return:
        """
        return state in self.finals.keys()

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next state with current state and action given.
        :param state: If a state is given, do action from that state.
        :param action: from action_space
        :return: a new state (or old if is invalid action)
        """

        # Get my position
        x, y = state if state else self.current_state

        # Do movement
        if action == self._actions.get('RIGHT_PROB'):
            rnd_number = self.np_random.uniform()

            if rnd_number > self.transitions[0]:
                x += 1
            else:
                y += 1

        elif action == self._actions.get('DOWN_PROB'):
            rnd_number = self.np_random.uniform()

            if rnd_number > self.transitions[0]:
                y += 1
            else:
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

        # Can we go to right?
        x_right = x + 1

        # Check if we are in a border of mesh
        if x_right < self.observation_space[0].n:
            possible_actions.append(self._actions.get('RIGHT_PROB'))
            possible_actions.append(self._actions.get('DOWN_PROB'))
        else:
            possible_actions.append(self._actions.get('DOWN'))

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
        del data['dynamic_action_space']

        return data
