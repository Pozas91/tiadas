# coding=utf-8
"""
Base class to define environments.

Action space is a discrete number. Actions are in the range [0, n).

Finals is a dictionary where the key is a position, and the value is a reward
vector as follows:
{
    state_1: reward,
    state_2: reward,
    ...
}

Obstacles is a frozenset of states: {state_1, state_2, ...}
"""

from copy import deepcopy
from typing import Iterable

import gym
from gym.utils import seeding

from models import Vector
from spaces import IterableDiscrete


class Environment(gym.Env):
    # Possible actions. To be set in descendant classes.
    _actions = dict()

    # Icons to render environments
    _icons = {
        'BLANK': ' ',
        'BLOCK': '■',
        'TREASURE': '$',
        'CURRENT': '☺',
        'ENEMY': '×',
        'HOME': 'µ',
        'FINAL': '$'
    }

    def __init__(self, observation_space: gym.spaces, default_reward: Vector, action_space: gym.spaces = None,
                 seed: int = None, initial_state: Iterable = None, obstacles: frozenset = None, finals: object = None):
        """
        :param default_reward: Default reward returned by the environment when 
                               a reward is not defined.
        :param seed: Initial seed. The same is used for _action_space,
                     observation_space, and random number generator
        :param initial_state: start position for all episodes.
        :param obstacles: inaccessible states.
        :param finals: terminal states for episodes.
        """

        # Set action space
        self._action_space = action_space if action_space else IterableDiscrete(len(self._actions))
        self._action_space.seed(seed=seed)

        # Create the observation space
        self.observation_space = observation_space
        self.observation_space.seed(seed=seed)

        # Prepare random seed
        self.np_random = None
        self.initial_seed = seed
        self.seed(seed=seed)

        # Set current environment position
        assert initial_state is None or self.observation_space.contains(initial_state)
        self.initial_state = initial_state
        self.current_state = self.initial_state

        # Set finals states
        self.finals = finals if finals else dict()

        # Set obstacles
        self.obstacles = obstacles if obstacles else frozenset()

        # Defaults
        self.default_reward = default_reward

    @property
    def actions(self) -> dict:
        """
        Return a dictionary with possible actions
        :return:
        """
        return self._actions

    @property
    def icons(self) -> dict:
        """
        Return a dictionary with possible icons
        :return:
        """
        return self._icons

    @property
    def action_space(self) -> gym.spaces:
        return self._action_space

    def step(self, action: int) -> (object, Vector, bool, dict):
        """
        Standard operation in gym environments. Performs the 'action' in the
        environment, returning the new position, the vector reward, a boolean
        value indicating if the reached position is final, and an optional dictionary
        with miscellaneous information.
        :param action:
        :return: 
        """
        raise NotImplemented

    def seed(self, seed: int = None) -> list:
        """
        Standard operation in gym environments. Generate seed
        :param seed:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed=seed)
        return [seed]

    def reset(self) -> object:
        """
        Standard operation in gym environments. Reset environment to a known
        initial position.
        :return:
        """
        self.current_state = self.initial_state
        return self.current_state

    def render(self, mode: str = 'human') -> None:
        """
        Standard operation in gym environments. Render the environment.
        :param mode:
        :return:
        """
        raise NotImplemented

    def next_state(self, action: int, state: object = None) -> object:
        """
        Calc next position with current position and action given. Default is 4-neighbors (UP, LEFT, DOWN, RIGHT)
        :param state: If a position is given, do action from that position.
        :param action: from action_space
        :return: a new position (or old if is invalid action)
        """
        raise NotImplemented

    def get_dict_model(self) -> dict:
        """
        :return: a dictionary with the environments parameters.
        """

        # Prepare a deepcopy to do not override original properties
        model = deepcopy(self)

        # Extract properties
        data = vars(model)

        # Prepare train_data
        data['default_reward'] = model.default_reward.tolist()

        # Clean Environment Data
        del data['_action_space']
        del data['observation_space']
        del data['np_random']
        del data['finals']
        del data['obstacles']

        return data

    def is_final(self, state: object = None) -> bool:
        """
        Return True if position given is terminal, False in otherwise.
        :return:
        """
        state = state if state else self.current_state
        return state in self.finals

    def states(self) -> set:
        """
        Return all possible states of this environment.
        :return:
        """
        raise NotImplemented

    def reachable_states(self, state: object, action: int) -> set:
        """
        Return all reachable states for pair (s, a) given.
        :param state:
        :param action:
        :return:
        """
        raise NotImplemented

    def transition_probability(self, state: object, action: int, next_state: object) -> float:
        """
        Return probability to reach `next_state` from `position` using `action`.

        In non-stochastic environments this return always 1.

        :param state: initial position
        :param action: action to do
        :param next_state: next position reached
        :return:
        """
        return 1.

    def transition_reward(self, state: object, action: int, next_state: object) -> object:
        """
        Return reward for reach `next_state` from `position` using `action`.

        :param state: initial position
        :param action: action to do
        :param next_state: next position reached
        :return:
        """
        raise NotImplemented
