"""
Inspired by the Deep Sea Treasure (DST) environment. In contrast to the, the values of the treasures are altered to
create a convex Pareto front.

FINAL STATE: To reach any final position.

REF: Multi-objective reinforcement learning using sets of pareto dominating policies (Kristof Van Moffaert,
Ann Nowé) 2014

HV REFERENCE: (-25, 0, -120)
"""
import gym

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

    # Experiments common hypervolume reference
    hv_reference = Vector([-25, 0, -120])

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0,), seed: int = 0, columns: int = 0,
                 action_space: gym.spaces = None):
        """
        :param initial_state:
        :param default_reward: (treasure_value)
        :param seed:
        """

        original_mesh_shape = (10, 11)

        # Reduce the number of diagonals
        if columns < 1 or columns > original_mesh_shape[0]:
            columns = original_mesh_shape[0]

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

        # Filter finals states
        finals = dict(filter(lambda x: x[0][0] < columns, finals.items()))

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

        # Filter obstacles states
        obstacles = frozenset(filter(lambda x: x[0] < columns, obstacles))

        # Resize mesh_shape
        mesh_shape = (columns, 11)

        # Default reward plus time (time_inverted, treasure_value, water_pressure)
        default_reward = (-1,) + default_reward + (0,)
        default_reward = Vector(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, default_reward=default_reward, initial_state=initial_state,
                         finals=finals, obstacles=obstacles, action_space=action_space)

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (position, (time_inverted, treasure_value), final, extra)
        """

        # Initialize reward as vector
        reward = self.default_reward.copy()

        # Update previous position
        self.current_state = self.next_state(action=action)

        # Get treasure value
        reward[1] = self.finals.get(self.current_state, self.default_reward[1])

        # Water pressure (y-coordinate)
        reward[2] = -(self.current_state[1] + 1)

        # Set extra
        info = {}

        # Check is_final
        final = self.is_final(self.current_state)

        return self.current_state, reward, final, info

    def transition_reward(self, state: tuple, action: int, next_state: tuple) -> Vector:
        # Default reward
        reward = self.default_reward.copy()

        # Get treasure reward
        reward[1] = self.finals.get(next_state, self.default_reward[1])

        # Water pressure (y-coordinate)
        reward[2] = -(next_state[1] + 1)

        return reward
