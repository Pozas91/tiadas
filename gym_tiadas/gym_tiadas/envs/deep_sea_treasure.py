"""The environment is a grid of 10 rows and 11 columns. The agent controls a submarine searching for undersea
treasure. There are multiple treasure locations with varying values. There are two objectives - to minimise the time
taken to reach the treasure, and to maximise the value of the treasure. Each episode commences with the vessel in the
top left state, and ends when a treasure location is reached or after 1000 actions. Four actions are available to the
agent - moving one square to the left, right, up or down. Any action which would cause the agent to leave the grid
will leave its position unchanged. The reward received by the agent is a 2-element vector. The first element is a
time penalty, which is -1 on all turns. The second element is the treasure value which is 0 except when the agent
moves into a treasure location, when it is the value indicated.

FINAL STATE: To reach any final state.

REF: Empirical Evaluation methods for multi-objective reinforcement learning algorithms
    (Vamplew, Dazeley, Berry, Issabekov and Dekker) 2011
"""
from .env_mesh import EnvMesh


class DeepSeaTreasure(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Pareto optimal
    pareto_optimal = [
        (-1, 1), (-3, 2), (-5, 3), (-7, 5), (-8, 8), (-9, 16), (-13, 24), (-14, 50), (-17, 74), (-19, 124)
    ]

    def __init__(self, initial_state=(0, 0), default_reward=0., seed=0):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
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

        mesh_shape = (10, 11)

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
        """
        Return True if state given is terminal, False in otherwise.
        :param state:
        :return:
        """
        return state in self.finals.keys()
