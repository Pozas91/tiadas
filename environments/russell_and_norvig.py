"""
Mesh problem with a 4x3 grid. We have an agent that try reached goal avoiding a trap. The environment has a p_stochastic
list of probabilities that can change agent's action to another.
"""

import utils.environments as ue
from models import VectorDecimal
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
        obstacles = obstacles.union({(1, 1)})

        # Default shape
        mesh_shape = (4, 3)
        default_reward = VectorDecimal(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, initial_state=initial_state, obstacles=obstacles,
                         finals=finals, default_reward=default_reward)

        self.transitions = transitions

    def step(self, action: int) -> (tuple, VectorDecimal, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Initialize reward as vector
        reward = self.default_reward.copy()

        # Get probability action
        action = self.__probability_action(action=action)

        # Update previous state
        self.current_state = self.next_state(action=action)

        # Get reward
        reward[0] = self.finals.get(self.current_state, self.default_reward[0])

        # Check if is final position
        final = self.is_final(self.current_state)

        # Set info
        info = {}

        return self.current_state, reward, final, info

    def __probability_action(self, action: int) -> int:
        """
        Decide probability action after apply probabilistic p_stochastic.
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

    def transition_reward(self, state: tuple, action: int, next_state: tuple) -> float:
        # Initialize reward as vector
        reward = self.default_reward.copy()
        reward[0] = self.finals.get(next_state, reward[0])

        return reward

    def transition_probability(self, state: tuple, action: int, next_state: tuple) -> float:

        n_actions = len(self.actions)
        coefficient = (n_actions - action)

        if ue.is_on_up(state=state, next_state=next_state):
            probability = self.transitions[(coefficient + 0) % n_actions]
        elif ue.is_on_right(state=state, next_state=next_state):
            probability = self.transitions[(coefficient + 1) % n_actions]
        elif ue.is_on_down(state=state, next_state=next_state):
            probability = self.transitions[(coefficient + 2) % n_actions]
        else:
            probability = self.transitions[(coefficient + 3) % n_actions]

        return probability

    def reachable_states(self, state: tuple, action: int) -> set:
        # Return all possible states reachable with any action
        return {self.next_state(action=a, state=state) for a in self.actions.values() if a != a + 2}
