"""
Mesh problem with a 4x3 grid. We have an agent that try reached goal avoiding a trap. The environment has a transitions
list of probabilities that can change agent's action to another.
"""
import math

import numpy as np

from models import VectorFloat
from .env_mesh import EnvMesh


class RussellNorvig(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    def __init__(self, transitions: tuple = (0.8, 0.1, 0., 0.1), initial_state: tuple = (0, 2), seed: int = 0,
                 default_reward: tuple = (-0.04,)):
        """
        :param transitions:
            Probabilities to change direction of action given.
            [DIR_0, DIR_90, DIR_180, DIR_270]
        :param initial_state:
        :param default_reward:
        """

        # finals states and its reward
        finals = {
            (3, 0): 1,
            (3, 1): -1
        }

        # Set of obstacles
        obstacles = frozenset()
        obstacles = obstacles.union([(1, 1)])

        # Default shape
        mesh_shape = (4, 3)
        default_reward = VectorFloat(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, initial_state=initial_state, obstacles=obstacles,
                         finals=finals, default_reward=default_reward)

        assert isinstance(transitions, tuple) and math.isclose(a=np.sum(transitions), b=1) and len(transitions) == len(
            self._actions)
        self.transitions = transitions

    def step(self, action: int) -> (tuple, VectorFloat, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Initialize rewards as vector
        reward = self.default_reward.copy()

        # Get probability action
        action = self.__probability_action(action=action)

        # Get new state
        new_state = self.next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Get reward
        reward[0] = self.finals.get(self.current_state, self.default_reward[0])

        # Check if is final state
        final = self.is_final(self.current_state)

        # Set info
        info = {}

        return new_state, reward, final, info

    def reset(self) -> tuple:
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        return self.current_state

    def __probability_action(self, action: int) -> int:
        """
        Decide probability action after apply probabilistic transitions.
        :param action:
        :return:
        """

        # Get a random uniform number [0., 1.]
        random = self.np_random.uniform()

        # Start with first direction
        direction = 0

        # Accumulate roulette
        roulette = self.transitions[direction]

        # While random is greater than roulette
        while random > roulette:
            # Increment action
            direction += 1

            # Increment roulette
            roulette += self.transitions[direction]

        # Cyclic direction
        return (direction + action) % self.action_space.n

    def is_final(self, state: tuple = None) -> bool:
        """
        Is final if agent is on final state.
        :param state:
        :return:
        """
        return state in self.finals.keys()
