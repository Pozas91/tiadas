"""
Such as DeepSeaTreasure environment but has a vector of transitions probabilities, which will be used when an action
is to be taken.

HV REFERENCE: (-25, 0)
"""
from models import Vector
from .env_mesh import EnvMesh


class DeepSeaTreasureStochastic(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Experiments common hypervolume reference
    hv_reference = Vector([-25, 0])

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0,), seed: int = 0,
                 n_transition: float = 0.3):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        :param n_transition: if is 0, the action affected by the transition is always the same action.
        """

        # List of all treasures and its reward.
        finals = {
            (0, 1): 1,
            (1, 2): 2,
            (2, 3): 3,
            (3, 4): 5,
            (4, 4): 8,
            (5, 4): 16,
            (6, 7): 24,
            (7, 7): 50,
            (8, 9): 74,
            (9, 10): 124,
        }

        mesh_shape = (10, 11)

        # Default reward plus time (time_inverted, treasure_value)
        default_reward = (-1,) + default_reward
        default_reward = Vector(default_reward)

        obstacles = frozenset()
        obstacles = obstacles.union([(0, y) for y in range(2, 11)])
        obstacles = obstacles.union([(1, y) for y in range(3, 11)])
        obstacles = obstacles.union([(2, y) for y in range(4, 11)])
        obstacles = obstacles.union([(3, y) for y in range(5, 11)])
        obstacles = obstacles.union([(4, y) for y in range(5, 11)])
        obstacles = obstacles.union([(5, y) for y in range(5, 11)])
        obstacles = obstacles.union([(6, y) for y in range(8, 11)])
        obstacles = obstacles.union([(7, y) for y in range(8, 11)])
        obstacles = obstacles.union([(8, y) for y in range(10, 11)])

        # Check transaction probability
        super().__init__(mesh_shape=mesh_shape, seed=seed, default_reward=default_reward, initial_state=initial_state,
                         finals=finals, obstacles=obstacles)

        # Transaction
        assert 0 <= n_transition <= 1.

        # [DIR_0, DIR_90, DIR_180, DIR_270] transaction tuple
        self.transitions = (1. - n_transition, n_transition / 3, n_transition / 3, n_transition / 3)

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (time_inverted, treasure_value), final, info)
        """

        # Get probability action
        action = self.__probability_action(action=action)

        # Initialize rewards as vector
        rewards = self.default_reward.copy()

        # Get new state
        new_state = self.next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Get treasure value
        rewards[1] = self.finals.get(self.current_state, self.default_reward[1])

        # Set info
        info = {}

        # Check is_final
        final = self.is_final(self.current_state)

        return self.current_state, rewards, final, info

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
