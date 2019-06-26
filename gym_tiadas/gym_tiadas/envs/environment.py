# coding=utf-8
"""
Base model to define environments.

Action space is a discrete number. Allow numbers from [0, n).

Finals is a dictionary which structure as follows:
{
    state_1: reward,
    state_2: reward,
    ...
}

Obstacles is a frozenset of states: {state_1, state_2, ...}
"""

from copy import deepcopy

import gym
from gym.utils import seeding

from models import Vector
from spaces import IterableDiscrete


class Environment(gym.Env):
    # Possible actions
    _actions = dict()

    # Icons to render environments
    _icons = {'BLANK': ' ', 'BLOCK': '■', 'TREASURE': '$', 'CURRENT': '☺', 'ENEMY': '×', 'HOME': 'µ', 'FINAL': '$'}

    def __init__(self, observation_space: gym.spaces, default_reward: Vector, seed: int = None,
                 initial_state: object = None, obstacles: frozenset = None, finals: dict = None):
        """
        :param default_reward: Default reward that return environment when a reward is not defined.
        :param seed: Initial seed.
        :param initial_state: First state where agent start.
        :param obstacles: States where agent can not to be.
        :param finals: States where agent finish an epoch.
        """

        # Set action space
        self._action_space = IterableDiscrete(len(self._actions))
        self._action_space.seed(seed=seed)

        # Create the observation space
        self.observation_space = observation_space

        # Prepare random seed
        self.np_random = None
        self.initial_seed = seed
        self.seed(seed=seed)

        # Set current environment state
        assert initial_state is None or self.observation_space.contains(initial_state)
        self.initial_state = initial_state
        self.current_state = self.initial_state

        # Set finals states
        assert finals is None or all([self.observation_space.contains(final) for final in finals.keys()])
        self.finals = finals

        # Set obstacles
        assert obstacles is None or all([self.observation_space.contains(obstacle) for obstacle in obstacles])
        self.obstacles = obstacles

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
        Do a step in the environment
        :param action:
        :return:
        """
        raise NotImplemented

    def seed(self, seed: int = None) -> list:
        """
        Generate seed
        :param seed:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed=seed)
        return [seed]

    def reset(self) -> object:
        """
        Reset environment to zero.
        :return:
        """
        raise NotImplemented

    def render(self, mode: str = 'human') -> None:
        """
        Render environment
        :param mode:
        :return:
        """
        raise NotImplemented

    def next_state(self, action: int, state: object = None) -> object:
        """
        Calc next state with current state and action given. Default is 4-neighbors (UP, LEFT, DOWN, RIGHT)
        :param state: If a state is given, do action from that state.
        :param action: from action_space
        :return: a new state (or old if is invalid action)
        """
        raise NotImplemented

    def get_dict_model(self) -> dict:
        """
        Get dict model of an environment
        :return:
        """

        # Prepare a deepcopy to do not override original properties
        model = deepcopy(self)

        # Extract properties
        data = vars(model)

        # Prepare data
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
        Return True if state given is terminal, False in otherwise.
        :return:
        """
        raise NotImplemented
