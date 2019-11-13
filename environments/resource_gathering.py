"""
An agent begins at the home location in a 2D grid, and can move one square at a time in each of the four cardinal
directions. The agent's task is to collect either or both of two resources (gold and gems) which are available at fixed
locations, and return home with these resources. The environment contains two locations at which an enemy attack may
occur, with a 10% probability. If an attack happens, the agent loses any resources currently being carried and is returned
to the home location. The reward vector is ordered as [enemy, gold, gems] and there are four possible rewards which may
be received on entering the home location.

• [−1, 0, 0] in case of an enemy attack;
• [0, 1, 0] for returning home with gold but no gems;
• [0, 0, 1] for returning home with gems but no gold;
• [0, 1, 1] for returning home with both gold and gems.

FINAL STATE: any of below states.

REF: Empirical Evaluation methods for multi-objective reinforcement learning algorithms
    (Vamplew, Dazeley, Berry, Issabekov and Dekker) 2011
"""
import itertools

import gym

from models import Vector
from .env_mesh import EnvMesh


class ResourceGathering(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    def __init__(self, initial_state: tuple = ((2, 4), (0, 0)), default_reward: tuple = (0, 0, 0), seed: int = 0,
                 p_attack: float = 0.1):
        """
        :param initial_state:
        :param default_reward: (enemy_attack, gold, gems)
        :param seed:
        :param p_attack: Probability that a enemy attacks when agent stay in an enemy position.
        """

        # Positions where there are gold {position: available}
        self.gold_positions = {(2, 0): True}

        # Positions where there is a gem {position: available}
        self.gem_positions = {(4, 1): True}

        mesh_shape = (5, 5)
        default_reward = Vector(default_reward)

        # Build the observation space (position(x, y), quantity(gold, gems))
        observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Tuple(
                    (gym.spaces.Discrete(mesh_shape[0]), gym.spaces.Discrete(mesh_shape[1]))
                ),
                gym.spaces.Tuple(
                    (gym.spaces.Discrete(2), gym.spaces.Discrete(2))
                )
            )
        )

        # Define final states
        finals = {
            ((2, 4), (1, 0)),
            ((2, 4), (0, 1)),
            ((2, 4), (1, 1)),
        }

        # Super constructor call.
        super().__init__(mesh_shape=mesh_shape, seed=seed, initial_state=initial_state, default_reward=default_reward,
                         observation_space=observation_space, finals=finals)

        # States where there are enemies_positions
        self.enemies_positions = {(3, 0), (2, 1)}

        self.p_attack = p_attack

        self.home_position = (2, 4)

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Initialize reward as vector
        reward = self.default_reward.copy()

        # Update previous position
        self.current_state, attacked = self.next_state(action=action)

        if attacked:
            reward[0] = -1

        # Check is_final
        final = attacked or self.is_final()

        if final:
            reward[1], reward[2] = self.current_state[1]

        # Set extra
        info = {}

        return self.current_state, reward, final, info

    def next_state(self, action: int, state: tuple = None) -> (tuple, bool):

        # Unpack complex state (position, objects(gold, gem))
        position, objects = state if state else self.current_state

        # Calc next position
        next_position, is_valid = self.next_position(action=action, position=position)

        # Extra information
        attacked = False

        # If the next_position isn't valid, reset to the previous position
        if not self.observation_space[0].contains(next_position) or not is_valid:
            next_position = position

        if next_position in self.gold_positions and self.gold_positions[next_position]:
            objects = 1, objects[1]
            self.gold_positions.update({next_position: False})

        elif next_position in self.gem_positions and self.gem_positions[next_position]:
            objects = objects[0], 1
            self.gem_positions.update({next_position: False})

        elif next_position in self.enemies_positions and self.p_attack >= self.np_random.uniform():
            next_position, objects = self.initial_state
            next_position = self.home_position
            attacked = True

        return (next_position, objects), attacked

    def reset(self) -> tuple:
        """
        Reset environment to zero.
        :return:
        """

        # Reset to initial seed
        self.seed(seed=self.initial_seed)

        # Reset golds positions
        for gold_state in self.gold_positions.keys():
            self.gold_positions.update({gold_state: True})

        # Reset gems positions
        for gem_state in self.gem_positions.keys():
            self.gem_positions.update({gem_state: True})

        self.current_state = self.initial_state
        return self.current_state

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment train_data
        del data['gold_positions']
        del data['gem_positions']
        del data['enemies_positions']
        del data['home_position']

        return data

    def states(self) -> set:

        # Unpack spaces
        x_position, y_position = self.observation_space[0]

        # Calc basic states
        basic_states = {
                           (x, y) for x in range(x_position.n) for y in range(y_position.n)
                       } - self.obstacles

        # Calc product of basic states with objects
        states = set(itertools.product(basic_states, {(0, 0), (0, 1), (1, 0), (1, 1)})).difference(self.finals)

        # Return all spaces
        return states

    # def transition_reward(self, state: tuple, action: int, next_state: tuple) -> tuple:
