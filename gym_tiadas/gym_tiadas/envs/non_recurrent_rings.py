"""
Such as LinkedRings environment, but in this case we have an extra state and different values.
"""

import gym
from gym import spaces
from gym.utils import seeding

from models import Vector


class NonRecurrentRings(gym.Env):
    # Possible actions
    _actions = {'CLOCKWISE': 0, 'COUNTER-CLOCKWISE': 1}

    # Icons to render environments
    _icons = {'BLANK': ' ', 'BLOCK': '■', 'TREASURE': '$', 'CURRENT': '☺', 'ENEMY': '×', 'HOME': 'µ', 'FINAL': '$'}

    def __init__(self, seed=0, initial_state=0):
        """

        :param seed:
        :param initial_state:
        """

        # Set action space
        self.action_space = spaces.Discrete(len(self.actions))

        # Create the observation space
        self.observation_space = spaces.Discrete(8)

        # Prepare random seed
        self.np_random = None
        self.seed(seed=seed)

        # Set current environment state
        assert initial_state is None or self.observation_space.contains(initial_state)
        self.initial_state = initial_state
        self.current_state = self.initial_state

        # Rewards dictionary
        self.rewards_dictionary = {
            0: {
                self.actions.get('COUNTER-CLOCKWISE'): (2, -1),
                self.actions.get('CLOCKWISE'): (-1, 0)
            },
            1: {
                self.actions.get('COUNTER-CLOCKWISE'): (2, -1),
                self.actions.get('CLOCKWISE'): (-1, 0)
            },
            2: {
                self.actions.get('COUNTER-CLOCKWISE'): (2, -1),
                self.actions.get('CLOCKWISE'): (-1, 0)
            },
            3: {
                self.actions.get('COUNTER-CLOCKWISE'): (2, -1),
                self.actions.get('CLOCKWISE'): (-1, 0)
            },
            4: {
                self.actions.get('CLOCKWISE'): (-1, 2),
                self.actions.get('COUNTER-CLOCKWISE'): (0, -1)
            },
            5: {
                self.actions.get('CLOCKWISE'): (-1, 2),
                self.actions.get('COUNTER-CLOCKWISE'): (0, -1)
            },
            6: {
                self.actions.get('CLOCKWISE'): (-1, 2),
                self.actions.get('COUNTER-CLOCKWISE'): (0, -1)
            },
            7: {
                self.actions.get('CLOCKWISE'): (-1, 2),
                self.actions.get('COUNTER-CLOCKWISE'): (0, -1)
            }
        }

        # Possible transactions from a state to another
        self.possible_transactions = {
            0: {
                self.actions.get('COUNTER-CLOCKWISE'): 1,
                self.actions.get('CLOCKWISE'): 7
            },
            1: {
                self.actions.get('COUNTER-CLOCKWISE'): 2,
                self.actions.get('CLOCKWISE'): 0
            },
            2: {
                self.actions.get('COUNTER-CLOCKWISE'): 3,
                self.actions.get('CLOCKWISE'): 1
            },
            3: {
                self.actions.get('COUNTER-CLOCKWISE'): 0,
                self.actions.get('CLOCKWISE'): 2
            },
            4: {
                self.actions.get('CLOCKWISE'): 5,
                self.actions.get('COUNTER-CLOCKWISE'): 7
            },
            5: {
                self.actions.get('CLOCKWISE'): 6,
                self.actions.get('COUNTER-CLOCKWISE'): 4
            },
            6: {
                self.actions.get('CLOCKWISE'): 7,
                self.actions.get('COUNTER-CLOCKWISE'): 5
            },
            7: {
                self.actions.get('CLOCKWISE'): 4,
                self.actions.get('COUNTER-CLOCKWISE'): 0
            }
        }

        # Reset environment
        self.reset()

    def step(self, action) -> (object, Vector, bool, dict):
        """
        Do a step in the environment
        :param action:
        :return:
        """

        # Get new state
        new_state = self.next_state(action=action)

        # Get reward
        reward = self.rewards_dictionary.get(self.current_state).get(action)

        # Update previous state
        self.current_state = new_state

        # Check if is final state
        final = self.is_final()

        # Set info
        info = {}

        return new_state, reward, final, info

    def seed(self, seed=None):
        """
        Generate seed
        :param seed:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed=seed)
        return [seed]

    def reset(self):
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        return self.current_state

    def render(self, mode='human'):
        """
        Render environment
        :param mode:
        :return:
        """

    def next_state(self, action) -> object:
        """
        Calc next state with current state and action given.
        :param action: from action_space
        :return: a new state (or old if is invalid action)
        """

        # Do movement
        new_state = self.possible_transactions.get(self.current_state).get(action)

        if not self.observation_space.contains(new_state):
            # New state is invalid, and roll back with previous.
            new_state = self.current_state

        # Return new state
        return new_state

    @property
    def actions(self):
        """
        Return a dictionary with possible actions
        :return:
        """
        return self._actions

    def is_final(self, state=None) -> bool:
        # (This is a non-episodic problem, so doesn't have final states)
        return False
