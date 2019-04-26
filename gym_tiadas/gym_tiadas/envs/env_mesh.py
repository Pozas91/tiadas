"""
Base model to define environments which observation_space as mesh.
Mesh is a grid of columns per rows, where the minimum value is (0, 0), and maximum value is (columns - 1, rows - 1).
Top-left corner is (0, 0), top-right corner is (columns - 1, 0), bottom-left corner is (0, rows - 1) and
bottom-right corner is (columns - 1, rows - 1).

Action space is a discrete number. Allow numbers from [0, n).

Finals is a dictionary which structure as follows:
{
    state_1: reward,
    state_2: reward,
    ...
}

Obstacles is a list of states: [state_1, state_2, ...]
"""

import gym
from gym import spaces
from gym.utils import seeding

from models import Vector


class EnvMesh(gym.Env):
    # Possible actions
    _actions = dict()

    # Icons to render environments
    _icons = {'BLANK': ' ', 'BLOCK': '■', 'TREASURE': '$', 'CURRENT': '☺', 'ENEMY': '×', 'HOME': 'µ', 'FINAL': '$'}

    def __init__(self, mesh_shape: tuple, default_reward=None, seed=None, initial_state=None, obstacles=None,
                 finals=None):

        # Set action space
        self.action_space = spaces.Discrete(len(self._actions))

        # Create the mesh
        x, y = mesh_shape
        self.observation_space = spaces.Tuple((spaces.Discrete(x), spaces.Discrete(y)))

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

        # Reset environment
        self.reset()

    @property
    def actions(self):
        """
        Return a dictionary with possible actions
        :return:
        """
        return self._actions

    @property
    def icons(self):
        """
        Return a dictionary with possible icons
        :return:
        """
        return self._icons

    def step(self, action) -> (object, Vector, bool, dict):
        """
        Do a step in the environment
        :param action:
        :return:
        """
        raise NotImplemented

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
        raise NotImplemented

    def render(self, mode='human'):
        """
        Render environment
        :param mode:
        :return:
        """

        if mode == 'human':
            # Get cols (x) and rows (y) from observation space
            cols, rows = self.observation_space.spaces[0].n, self.observation_space.spaces[1].n

            for y in range(rows):
                for x in range(cols):

                    # Set a state
                    state = (x, y)

                    if state == self.current_state:
                        icon = self._icons.get('CURRENT')
                    elif state in self.obstacles:
                        icon = self._icons.get('BLOCK')
                    elif state in self.finals.keys():
                        icon = self.finals.get(state)
                    else:
                        icon = self._icons.get('BLANK')

                    # Show col
                    print('| {} '.format(icon), end='')

                # New row
                print('|')

            # End render
            print('')

    def _next_state(self, action) -> tuple:
        """
        Calc next state with current state and action given. Default is 4-neighbors (UP, LEFT, DOWN, RIGHT)
        :param action: from action_space
        :return: a new state (or old if is invalid action)
        """

        # Get my position
        x, y = self.current_state

        # Do movement
        if action == self._actions.get('UP'):
            y -= 1
        elif action == self._actions.get('RIGHT'):
            x += 1
        elif action == self._actions.get('DOWN'):
            y += 1
        elif action == self._actions.get('LEFT'):
            x -= 1

        # Set new state
        new_state = x, y

        # If exists obstacles, then new_state must be in self.obstacles
        is_obstacle = bool(self.obstacles) and new_state in self.obstacles

        if not self.observation_space.contains(new_state) or is_obstacle:
            # New state is invalid.
            new_state = self.current_state

        # Return (x, y) position
        return new_state

    def get_dict_model(self):
        """
        Get dict model of an environment
        :return:
        """
        data = vars(self)

        # Prepare data
        data['default_reward'] = self.default_reward.tolist()

        # Clean Environment Data
        del data['action_space']
        del data['observation_space']
        del data['np_random']
        del data['finals']
        del data['obstacles']

        return data

    def is_final(self, state=None) -> bool:
        """
        Return True if state given is terminal, False in otherwise.
        :return:
        """
        raise NotImplemented
