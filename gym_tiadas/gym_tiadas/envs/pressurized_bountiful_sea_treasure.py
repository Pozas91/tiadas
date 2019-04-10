"""
Inspired by the Deep Sea Treasure (DST) environment. In contrast to the, the values of the treasures are altered to
create a convex Pareto front.

FINAL STATE: To reach any final state.

REF: Multi-objective reinforcement learning using sets of pareto dominating policies (Kristof Van Moffaert,
Ann Nowé) 2014 """
from .env_mesh import EnvMesh


class PressurizedBountifulSeaTreasure(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Pareto optimal
    pareto_optimal = [
        (-1, 5), (-3, 80), (-5, 120), (-7, 120), (-8, 145), (-9, 150), (-13, 163), (-14, 166), (-17, 173), (-19, 175)
    ]

    def __init__(self, mesh_shape=(10, 11), initial_observation=(0, 0), default_reward=0., seed=0):
        """
        :param initial_observation:
        :param default_reward:
        :param seed:
        """

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

        super().__init__(mesh_shape, seed, default_reward=default_reward, initial_state=initial_observation,
                         finals=finals, obstacles=obstacles)

    def step(self, action) -> (object, [float, float, float], bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (time_inverted, treasure_value), final, info)
        """

        # (time_inverted, treasure_value, water_pressure)
        rewards = [0., 0., 0.]

        # Get new state
        new_state = self._next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Get time inverted
        rewards[0] = -1

        # Get treasure value
        rewards[1] = self.finals.get(self.current_state, self.default_reward)

        # Water pressure (y-coordinate)
        rewards[2] = -(self.current_state[1] + 1)

        # Set info
        info = {}

        # Check is_final
        final = self.is_final(self.current_state)

        return self.current_state, rewards, final, info

    def reset(self):
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state

        return self.current_state

    def is_final(self, state=None) -> bool:
        # If agent is in treasure
        return state in self.finals.keys()
