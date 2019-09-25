"""
Inspired by the Deep Sea Treasure (DST) environment. In contrast to the, the values of the treasures are altered to
create a convex Pareto front.

FINAL STATE: To reach any final state.

REF: Multi-objective reinforcement learning using sets of pareto dominating policies (Kristof Van Moffaert,
Ann Nowé) 2014

HV REFERENCE: (-25, 0, -120)
"""
from models import Vector
from .env_mesh import EnvMesh


class PressurizedBountifulSeaTreasure(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Pareto optimal
    pareto_optimal = [
        (-1, 5, -2), (-3, 80, -3), (-5, 120, -4), (-7, 140, -5), (-8, 145, -6), (-9, 150, -6), (-13, 163, -8),
        (-14, 166, -8), (-17, 173, -10), (-19, 175, -11)
    ]

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0,), seed: int = 0):
        """
        :param initial_state:
        :param default_reward: (treasure_value)
        :param seed:
        """

        mesh_shape = (10, 11)

        # List of all treasures and its reward.
        finals = {
            (0, 1): 5,
            (1, 2): 80,
            (2, 3): 120,
            (3, 4): 140,
            (4, 4): 145,
            (5, 4): 150,
            (6, 7): 163,
            (7, 7): 166,
            (8, 9): 173,
            (9, 10): 175,
        }

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

        # Default reward plus time (time_inverted, treasure_value, water_pressure)
        default_reward = (-1,) + default_reward + (0,)
        default_reward = Vector(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, default_reward=default_reward,
                         initial_state=initial_state, finals=finals, obstacles=obstacles)

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (time_inverted, treasure_value), final, info)
        """

        # Initialize rewards as vector
        rewards = self.default_reward.copy()

        # Get new state
        new_state = self.next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Get treasure value
        rewards[1] = self.finals.get(self.current_state, self.default_reward[1])

        # Water pressure (y-coordinate)
        rewards[2] = -(self.current_state[1] + 1)

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

    def is_final(self, state: tuple = None) -> bool:
        """
        If agent is in treasure
        :param state:
        :return:
        """
        return state in self.finals.keys()
