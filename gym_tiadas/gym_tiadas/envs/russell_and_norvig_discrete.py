import gym
import numpy as np

from gym import spaces
from gym.utils import seeding


class RussellNorvigDiscrete(gym.Env):
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

        if not isinstance(mesh_shape, tuple):
            mesh_shape = 4, 3

        self.action_space = spaces.Discrete(len(self.__actions))
        self.observation_space = spaces.Discrete(mesh_shape[0] * mesh_shape[1])
        self.shape = mesh_shape
        self.default_reward = default_reward

        # Prepare finals states
        if finals is None:
            finals = {3: 1., 7: -1.}
        else:
            for key, value in finals.items():
                finals[self.__tuple_to_discrete(t=key)] = finals.pop(key)

        # Prepare to set initial observation
        initial_observation = self.__tuple_to_discrete(initial_observation) if isinstance(initial_observation,
                                                                                          tuple) else 8

        # Prepare obstacles
        obstacles = [5] if obstacles is None else [self.__tuple_to_discrete(t=t) for t in obstacles]

        # Prepare transactions
        if transactions is None:
            transactions = [0.8, 0.1, 0.0, 0.1]

        # Check initial observation
        assert self.observation_space.contains(initial_observation)
        self.initial_observation = initial_observation
        self.current_observation = self.initial_observation

        # Check obstacles
        assert isinstance(obstacles, list) and [self.observation_space.contains(obstacle) for obstacle in obstacles]
        self.obstacles = obstacles

        # Check finals
        assert isinstance(finals, dict) and [self.observation_space.contains(final) for final in finals.keys()]
        self.finals = finals

        # Check transactions
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
        cols, rows = self.shape

        for y in range(rows):
            for x in range(cols):

                # Set a state
                state = self.__tuple_to_discrete(t=(x, y))

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

    def __next_state(self, action) -> int:
        """
        Calc increment or decrement of state, if the new state is out of mesh, or is obstacle, return same state.
        :param action: UP, RIGHT, DOWN, LEFT
        :return:
        """

        # Get my position
        next_state = self.current_observation

        # Get shape
        x, y = self.shape

        mod_x = next_state % x

        # Check outbounds
        is_outbound_top = next_state < x
        is_outbound_right = mod_x == (x - 1)
        is_outbound_bottom = (((x - 1) * y) - 1) <= next_state <= ((x * y) - 1)
        is_outbound_left = mod_x == 0

        # Do movement (Outbound checking)
        if action == self.__actions.get('UP') and not is_outbound_top:
            next_state -= x
        elif action == self.__actions.get('RIGHT') and not is_outbound_right:
            next_state += 1
        elif action == self.__actions.get('DOWN') and not is_outbound_bottom:
            next_state += x
        elif action == self.__actions.get('LEFT') and not is_outbound_left:
            next_state -= 1

        if not self.observation_space.contains(next_state) or next_state in self.obstacles:
            # New state is invalid.
            next_state = self.current_observation

        return next_state

    def __tuple_to_discrete(self, t):
        """
        Convert the tuple given to discrete space
        :param t: Tuple (x, y)
        :return:
        """
        return t[1] * self.shape[0] + t[0]
