import gym
import numpy as np

from gym import spaces
from gym.utils import seeding


class RussellNorvig(gym.Env):
    __actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}
    __icons = {'BLANK': ' ', 'BLOCK': '■', 'FINAL': '$', 'CURRENT': '☺'}

    def __init__(self, mesh_shape=None, finals=None, obstacles=None, transactions=None, initial_observation=(0, 2),
                 default_reward=0., seed=0):
        """

        :param mesh_shape:
        :param finals:
        :param obstacles:
        :param transactions: [DIR_0, DIR_90, DIR_180, DIR_270]
        :param initial_observation:
        :param default_reward:
        """

        assert isinstance(mesh_shape, tuple) or mesh_shape is None

        if mesh_shape is tuple:
            x, y = mesh_shape
        else:
            x, y = 4, 3

        self.action_space = spaces.Discrete(len(self.__actions))
        self.observation_space = spaces.Tuple((spaces.Discrete(x), spaces.Discrete(y)))
        self.default_reward = default_reward

        if finals is None:
            finals = {
                (3, 0): 1.,
                (3, 1): -1.
            }

        if obstacles is None:
            obstacles = [(1, 1)]

        if transactions is None:
            transactions = [0.8, 0.1, 0.0, 0.1]

        assert isinstance(initial_observation, tuple) and self.observation_space.contains(initial_observation)
        self.initial_observation = initial_observation
        self.current_observation = self.initial_observation

        assert isinstance(obstacles, list) and [self.observation_space.contains(obstacle) for obstacle in obstacles]
        self.obstacles = obstacles

        assert isinstance(finals, dict) and [self.observation_space.contains(final) for final in finals.keys()]
        self.finals = finals

        assert isinstance(transactions, list) and np.sum(transactions) == 1. and len(transactions) == len(
            self.__actions)
        self.transactions = transactions

        self.reset()

        self.np_random = None
        self.seed(seed=seed)

    def seed(self, seed=None):
        """
        Generate seed
        :param seed:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed=seed)
        return [seed]

    def step(self, action) -> (object, float, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Get probability action
        action = self.__probability_action(action=action)

        # Get new state
        new_state = self.__next_state(action=action)

        # Update previous state
        self.current_observation = new_state

        # Get reward
        reward = self.finals.get(new_state, self.default_reward)

        # Check if is final state
        final = new_state in self.finals.keys()

        # Set info
        info = {}

        return new_state, reward, final, info

    def reset(self):
        self.current_observation = self.initial_observation
        return self.current_observation

    def render(self, **kwargs):

        # Get cols (x) and rows (y) from observation space
        cols, rows = self.observation_space.spaces[0].n, self.observation_space.spaces[1].n

        for y in range(rows):
            for x in range(cols):

                # Set a state
                state = (x, y)

                if state in self.obstacles:
                    icon = self.__icons.get('BLOCK')
                elif state in self.finals.keys():
                    icon = self.__icons.get('FINAL')
                elif state == self.current_observation:
                    icon = self.__icons.get('CURRENT')
                else:
                    icon = self.__icons.get('BLANK')

                # Show col
                print('| {} '.format(icon), end='')

            # New row
            print('|')

        # End render
        print('')

    def __probability_action(self, action) -> int:
        """
        Decide probability action after apply probabilistic transactions.
        :param action:
        :return:
        """

        # Get a random uniform number [0., 1.]
        random = self.np_random.uniform()

        # Start with first direction
        direction = self.__actions.get('UP')

        # Accumulate roulette
        roulette = self.transactions[direction]

        # While random is greater than roulette
        while random > roulette:
            # Increment action
            direction += 1

            # Increment roulette
            roulette += self.transactions[direction]

        # Cyclic direction
        return (direction + action) % self.action_space.n

    def __next_state(self, action) -> (int, int):
        """
        Calc increment or decrement of state, if the new state is out of mesh, or is obstacle, return same state.
        :param action: UP, RIGHT, DOWN, LEFT
        :return: x, y
        """

        # Get my position
        x, y = self.current_observation

        # Do movement
        if action == self.__actions.get('UP'):
            y -= 1
        elif action == self.__actions.get('RIGHT'):
            x += 1
        elif action == self.__actions.get('DOWN'):
            y += 1
        else:
            x -= 1

        # Set new state
        new_state = x, y

        if not self.observation_space.contains(new_state) or new_state in self.obstacles:
            # New state is invalid.
            new_state = self.current_observation

        # Return (x, y) position
        return new_state
