"""
Mesh problem with a 4x3 grid. We have an agent that try reached goal avoiding a trap. The environment has a transactions
list of probabilities that can change agent's action to another.
"""
import numpy as np

from .env_mesh import EnvMesh


class RussellNorvig(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    def __init__(self, transactions=(0.8, 0.1, 0., 0.1), initial_state=(0, 2),
                 default_reward=-0.04, seed=0):
        """
        :param transactions:
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

        # List of obstacles
        obstacles = [(1, 1)]

        # Default shape
        mesh_shape = (4, 3)

        super().__init__(mesh_shape, seed, initial_state=initial_state, obstacles=obstacles, finals=finals,
                         default_reward=default_reward)

        assert isinstance(transactions, tuple) and np.sum(transactions) == 1. and len(transactions) == len(
            self._actions)
        self.transactions = transactions

    def step(self, action) -> (object, float, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Get probability action
        action = self.__probability_action(action=action)

        # Get new state
        new_state = self._next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Get reward
        reward = self.finals.get(self.current_state, self.default_reward)

        # Check if is final state
        final = self.is_final(self.current_state)

        # Set info
        info = {}

        return new_state, reward, final, info

    def reset(self):
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        return self.current_state

    def __probability_action(self, action) -> int:
        """
        Decide probability action after apply probabilistic transactions.
        :param action:
        :return:
        """

        # Get a random uniform number [0., 1.]
        random = self.np_random.uniform()

        # Start with first direction
        direction = self._actions.get('UP')

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

    def is_final(self, state=None) -> bool:
        """
        Is final if agent is on final state.
        :param state:
        :return:
        """
        return state in self.finals.keys()
