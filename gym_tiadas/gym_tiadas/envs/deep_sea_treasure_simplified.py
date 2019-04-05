"""
DeepSeaTreasure environment simplified to test models.
"""
from .env_mesh import EnvMesh


class DeepSeaTreasureSimplified(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Pareto optimal
    pareto_optimal = [
        (-1, 5), (-3, 80), (-5, 120)
    ]

    def __init__(self, mesh_shape=(3, 4), initial_state=(0, 0), default_reward=0., seed=0):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        """

        # List of all treasures and its reward.
        finals = {
            (0, 1): 5,
            (1, 2): 80,
            (2, 3): 120,
        }

        obstacles = frozenset()
        obstacles = obstacles.union([(0, y) for y in range(2, 4)])
        obstacles = obstacles.union([(1, y) for y in range(3, 4)])

        super().__init__(mesh_shape, seed, initial_state=initial_state, default_reward=default_reward, finals=finals,
                         obstacles=obstacles)

    def step(self, action) -> (object, [float, float], bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (time_inverted, treasure_value), final, info)
        """

        # (time_inverted, treasure_value)
        rewards = [0., 0.]

        # Get new state
        new_state = self._next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Get time inverted
        rewards[0] = -1

        # Get treasure value
        rewards[1] = self.finals.get(self.current_state, self.default_reward)

        # Set info
        info = {}

        # If agent is in treasure
        final = self.current_state in self.finals.keys()

        return self.current_state, rewards, final, info

    def reset(self):
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state

        return self.current_state
