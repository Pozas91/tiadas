"""
Simple deterministic bi-objective non-episodic environment.

STATE SPACE:
-----------
The space state consists of 7 states linked by actions in a two ring shape.
State 1 is common to both rings.

     S2      S5
   /    \  /   \
S3       S1     S6
   \    / \     /
     S4      S7

All arcs are bidirectional, except  S4 -> S1 and S7 -> S1.
Therefore, the agent has two actions available at each state,

* Go to clockwise
* Go to counter-clockwise

States are implemented in a discrete 0-6 range as follows,

     1       4
   /    \  /   \
 2       0      5
   \    / \    /
     3       6


REWARDS:
-------
Left ring: (3,-1) moving counter-clockwise, (-1,0) otherwise.
Right ring: (-1,3) moving clockwise, (0, -1) otherwise.

FINAL STATE:
-----------
It is not an episodic task. Does not have finals states.

Reference:
Steering approaches to Pareto-optimal multiobjective reinforcement learning.
Peter Vamplew, Rustam Issabekov, Richard Dazeley, Cameron Foale, Adam Berry, 
Tim Moore, Douglas Creighton
Neurocomputing 263 (2017) 26-38.
"""

import gym
from gym import spaces
from gym.utils import seeding


class LinkedRings(gym.Env):
    # Possible actions
    _actions = {'CLOCKWISE': 0, 'COUNTER-CLOCKWISE': 1}

    # Icons to render environments
    _icons = {'BLANK': ' ', 'BLOCK': '■', 'TREASURE': '$', 'CURRENT': '☺',
              'ENEMY': '×', 'HOME': 'µ', 'FINAL': '$'}

    def __init__(self, seed=0, initial_state=0):
        """
        :param seed:
        :param initial_state:
        """

        # Set action space
        self.action_space = spaces.Discrete(len(self._actions))

        # Create the observation space
        self.observation_space = spaces.Discrete(7)

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
                self._actions.get('COUNTER-CLOCKWISE'): (3, -1),
                self._actions.get('CLOCKWISE'): (-1, 3)
            },
            1: {
                self._actions.get('COUNTER-CLOCKWISE'): (3, -1),
                self._actions.get('CLOCKWISE'): (-1, 0)
            },
            2: {
                self._actions.get('COUNTER-CLOCKWISE'): (3, -1),
                self._actions.get('CLOCKWISE'): (-1, 0)
            },
            3: {
                self._actions.get('COUNTER-CLOCKWISE'): (3, -1),
                self._actions.get('CLOCKWISE'): (-1, 0)
            },
            4: {
                self._actions.get('CLOCKWISE'): (-1, 3),
                self._actions.get('COUNTER-CLOCKWISE'): (0, -1)
            },
            5: {
                self._actions.get('CLOCKWISE'): (-1, 3),
                self._actions.get('COUNTER-CLOCKWISE'): (0, -1)
            },
            6: {
                self._actions.get('CLOCKWISE'): (-1, 3),
                self._actions.get('COUNTER-CLOCKWISE'): (0, -1)
            }
        }

        # Possible transitions from a state to another
        self.possible_transitions = {
            0: {
                self._actions.get('COUNTER-CLOCKWISE'): 1,
                self._actions.get('CLOCKWISE'): 4
            },
            1: {
                self._actions.get('COUNTER-CLOCKWISE'): 2,
                self._actions.get('CLOCKWISE'): 0
            },
            2: {
                self._actions.get('COUNTER-CLOCKWISE'): 3,
                self._actions.get('CLOCKWISE'): 1
            },
            3: {
                self._actions.get('COUNTER-CLOCKWISE'): 0,
                self._actions.get('CLOCKWISE'): 2
            },
            4: {
                self._actions.get('CLOCKWISE'): 5,
                self._actions.get('COUNTER-CLOCKWISE'): 0
            },
            5: {
                self._actions.get('CLOCKWISE'): 6,
                self._actions.get('COUNTER-CLOCKWISE'): 4
            },
            6: {
                self._actions.get('CLOCKWISE'): 0,
                self._actions.get('COUNTER-CLOCKWISE'): 5
            }
        }

        # Reset environment
        self.reset()

    def step(self, action):
        """
        Take a step in the environment
        :param action:
        :return:
        """

        # Get new state
        new_state = self.next_state(action=action)

        # Get reward
        reward = self.rewards_dictionary.get(self.current_state).get(action)

        # Update previous state
        self.current_state = new_state

        # Check is_final
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
        new_state = self.possible_transitions.get(self.current_state).get(action)

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
        """
        Checks if this is final state. 
        :param state:
        :return: Always False, since this task is not episodic.
        """
        return False
