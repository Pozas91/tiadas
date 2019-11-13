"""
PuddleWorld is a two-dimensional environment. The agent starts each episode at a random, non-goal position and has to move
to the goal in the top-right corner of the world, while avoiding the puddles. At each step selects between four actions
(left, right, up or down) which move it by 1 desired direction.
The mesh is 20x20 grid.
The reward structure for PuddleWorld is a tuple of two elements. First element is a penalty if goal is not reached, and
second element is a penalize by stay in a puddle, the penalize is the nearest distance to an edge of the puddle.

FINAL STATE: To reach (19, 0) position.

REF: Empirical evaluation methods for multi-objective reinforcement learning algorithms (2011).
"""
import gym
from scipy.spatial import distance

from models import VectorDecimal, Vector
from .env_mesh import EnvMesh


class MoPuddleWorld(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Experiments common hypervolume reference
    hv_reference = Vector([-50, -150])

    def __init__(self, default_reward: tuple = (10, 0), penalize_non_goal: float = -1, seed: int = 0,
                 final_state: tuple = (19, 0), action_space: gym.spaces = None):
        """
        :param default_reward: (non_goal_reached, puddle_penalize)
        :param penalize_non_goal: While agent does not reach a final position get a penalize.
        :param seed:
        :param final_state: This environment only has a final position.
        """

        self.final_state = final_state
        mesh_shape = (20, 20)
        default_reward = VectorDecimal(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, default_reward=default_reward, action_space=action_space)

        self.puddles = frozenset()
        self.puddles = self.puddles.union([(x, y) for x in range(0, 11) for y in range(3, 7)])
        self.puddles = self.puddles.union([(x, y) for x in range(6, 10) for y in range(2, 14)])
        self.penalize_non_goal = penalize_non_goal

        self.current_state = self.reset()

        # Get free spaces
        self.free_spaces = set(self.states() - self.puddles)

    def step(self, action: int) -> (tuple, VectorDecimal, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (position, (non_goal_reached, puddle_penalize), final, extra)
        """

        # Initialize reward as vector
        reward = self.default_reward.copy()

        # Update previous position
        self.current_state = self.next_state(action=action)

        # If agent is in treasure
        final = self.is_final(self.current_state)

        # Set final reward
        if not final:
            reward[0] = self.penalize_non_goal

        # if the current position is in an puddle
        if self.current_state in self.puddles:
            # Set penalization per distance
            reward[1] = self.calc_puddle_penalization(state=self.current_state)

        # Set extra
        info = {}

        return self.current_state, reward, final, info

    def calc_puddle_penalization(self, state: tuple):
        # Min distance found!
        min_distance = min(distance.cityblock(self.current_state, state) for state in self.free_spaces)

        # Set penalization per distance
        return -min_distance

    def reset(self) -> tuple:
        """
        Get random non-goal position to current_value
        :return:
        """

        # Reset to initial seed
        self.seed(seed=self.initial_seed)

        random_space = None

        while random_space == self.final_state:
            random_space = self.observation_space.sample()

        self.current_state = random_space
        return self.current_state

    def is_final(self, state: tuple = None) -> bool:
        """
        Is final if agent is on final position
        :param state:
        :return:
        """
        return state == self.final_state

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment train_data
        del data['puddles']
        del data['initial_state']

        return data

    def transition_reward(self, state: tuple, action: int, next_state: tuple) -> Vector:

        # Initialize reward as vector
        reward = self.default_reward.copy()

        # If agent is in treasure
        final = self.is_final(next_state)

        # Set final reward
        if not final:
            reward[0] = self.penalize_non_goal

        # if the current position is in an puddle
        if next_state in self.puddles:
            # Min distance found!
            min_distance = min(distance.cityblock(next_state, state) for state in self.free_spaces)

            # Set penalization per distance
            reward[1] = -min_distance

        return reward

    def states(self) -> set:

        # Unpack spaces
        x_position, y_position = self.observation_space.spaces

        return {
                   (x, y) for x in range(x_position.n) for y in range(y_position.n)
               } - {self.final_state}
