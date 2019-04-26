"""
PuddleWorld is a two-dimensional environment. The agent starts each episode at a random, non-goal state and has to move
to the goal in the top-right corner of the world, while avoiding the puddles. At each step selects between four actions
(left, right, up or down) which move it by 1 desired direction.
The mesh is 20x20 grid.
The reward structure for PuddleWorld is a tuple of two elements. First element is a penalty if goal is not reached, and
second element is a penalize by stay in a puddle, the penalize is the nearest distance to an edge of the puddle.

FINAL STATE: To reach (19, 0) state.

REF: Empirical evaluation methods for multi-objective reinforcement learning algorithms (2011).
"""
from scipy.spatial import distance

from models import VectorFloat
from .env_mesh import EnvMesh


class MoPuddleWorld(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    def __init__(self, default_reward=(10, 0), penalize_non_goal=-1, seed=0, final_state=(19, 0)):
        """

        :param default_reward: (non_goal_reached, puddle_penalize)
        :param penalize_non_goal:
        :param seed:
        :param final_state:
        """

        self.final_state = final_state
        mesh_shape = (20, 20)
        default_reward = VectorFloat(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, default_reward=default_reward)

        self.puddles = frozenset()
        self.puddles = self.puddles.union([(x, y) for x in range(0, 11) for y in range(3, 7)])
        self.puddles = self.puddles.union([(x, y) for x in range(6, 10) for y in range(2, 14)])
        self.penalize_non_goal = penalize_non_goal

        self.current_state = self.reset()

    def step(self, action) -> (object, VectorFloat, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (non_goal_reached, puddle_penalize), final, info)
        """

        # Initialize rewards as vector (plus zero to fast copy)
        rewards = self.default_reward + 0

        # Get new state
        new_state = self._next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # If agent is in treasure
        final = self.is_final(self.current_state)

        # Set final reward
        if not final:
            rewards[0] = self.penalize_non_goal

        # if the current state is in an puddle
        if self.current_state in self.puddles:
            # Unpack spaces
            x_space, y_space = self.observation_space.spaces
            # Get all spaces
            all_space = [(x, y) for x in range(x_space.n) for y in range(y_space.n)]
            # Get free spaces
            free_spaces = list(set(all_space) - self.puddles)
            # Start with infinite distance
            min_distance = float('inf')

            # For each free space
            for state in free_spaces:
                min_distance = min(min_distance, distance.cityblock(self.current_state, state))

            # Set penalization per distance
            rewards[1] = -min_distance

        # Set info
        info = {}

        return self.current_state, rewards, final, info

    def reset(self):
        """
        Get random non-goal state to current_value
        :return:
        """

        while True:
            random_space = self.observation_space.sample()

            if random_space != self.final_state:
                break

        self.current_state = random_space
        return self.current_state

    def is_final(self, state=None) -> bool:
        """
        Is final if agent is on final state
        :param state:
        :return:
        """
        return state == self.final_state

    def get_dict_model(self):
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment data
        del data['puddles']
        del data['initial_state']

        return data
