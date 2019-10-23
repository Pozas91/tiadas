# coding=utf-8
"""Like DST it is a 2D episodic grid environment, but it has three objectives rather than two. Each episode starts
with the agent in the location marked ’S’. The agent can move in the four cardinal directions, and receives a reward
of −1 for the time objective on every time-step. When reaching a terminal position the agent receives the rewards
specified in that cell for the other two objectives. In addition the rewards in the terminal states are doubled in
magnitude if the agent has activated the bonus by visiting the cell marked ’X2’. The black cells near the bonus
indicate walls which the agent cannot pass through. Similarly the agent cannot leave the bounds of the grid. Finally
the cells marked ’PIT’ indicate pits – if the agent enters one of these cells the bonus is deactivated, and the agent
returns to the start position. A tabular representation of this environment has 162 discrete states – 81 for the cells
of the grid when the agent has not activated the bonus, and 81 for the same cells when the bonus has been activated.
The set of Pareto-optimal policies and the corresponding thresholds are listed in Table 2. It can be seen that
trade-offs exist between all three objectives. Note that not all optimal policies require the agent to activate the
bonus.

FINAL STATE: To reach a final position.

REF: Vamplew et al (2017b)"""
from itertools import product

import gym

import spaces
from models import Vector
from .env_mesh import EnvMesh


class BonusWorld(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Experiments common hypervolume reference
    hv_reference = Vector([0, 0, -150])

    def __init__(self, initial_state: tuple = ((0, 0), False), default_reward: tuple = (0, 0), seed: int = 0,
                 action_space: gym.spaces = None):
        """
        :param initial_state:
        :param default_reward: (objective 1, objective 2)
        :param seed:
        """

        # List of all treasures and its reward.
        finals = {
            (8, 0): Vector([1, 9]),
            (8, 2): Vector([3, 9]),
            (8, 4): Vector([5, 9]),
            (8, 6): Vector([7, 9]),
            (8, 8): Vector([9, 9]),

            (0, 8): Vector([9, 1]),
            (2, 8): Vector([9, 3]),
            (4, 8): Vector([9, 5]),
            (6, 8): Vector([9, 7]),
        }

        # Define mesh shape
        mesh_shape = (9, 9)

        # Set obstacles
        obstacles = frozenset({(2, 2), (2, 3), (3, 2)})

        # Default reward plus time (objective 1, objective 2, time)
        default_reward += (-1,)
        default_reward = Vector(default_reward)

        # Build the observation space (position (x, y), bonus)
        observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Tuple(
                    (gym.spaces.Discrete(mesh_shape[0]), gym.spaces.Discrete(mesh_shape[1]))
                ),
                spaces.Boolean()
            )
        )

        super().__init__(mesh_shape=mesh_shape, default_reward=default_reward, initial_state=initial_state,
                         finals=finals, obstacles=obstacles, observation_space=observation_space, seed=seed,
                         action_space=action_space)

        # Pits marks which returns the agent to the start location.
        self.pits = {
            (7, 1), (7, 3), (7, 5), (1, 7), (3, 7), (5, 7)
        }

        # X2 bonus
        self.bonus = [
            (3, 3)
        ]

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (position, (objective 1, objective 2, time), final, info)
        """

        # Initialize reward as vector
        reward = self.default_reward.copy()

        # Unpack next position for reward
        position, bonus = self.next_state(action=action)

        # Get treasure value
        reward[0], reward[1] = self.finals.get(position, (self.default_reward[0], self.default_reward[1]))

        # If the bonus is activated, double the reward.
        if bonus:
            reward[0] *= 2
            reward[1] *= 2

        # Set info
        info = {}

        # Update current position
        self.current_state = position, bonus

        # Check is_final
        final = self.is_final(self.current_state)

        return self.current_state, reward, final, info

    def next_state(self, action: int, state: tuple = None) -> tuple:

        # Unpack complex position (position, bonus_activated)
        position, bonus = state if state else self.current_state

        # Calc next position
        next_position, is_valid = self.next_position(action=action, position=position)

        # If the next_position isn't valid, reset to the previous position
        if not self.observation_space[0].contains(next_position) or not is_valid:
            next_position = position

        # If agent is in pit, it's returned at initial position and deactivate the bonus.
        if next_position in self.pits:
            next_position, bonus = self.initial_state
            bonus = False

        # Check if the agent has activated the bonus
        elif next_position in self.bonus:
            bonus = True

        # Build next position
        return next_position, bonus

    def is_final(self, state: tuple = None) -> bool:
        """
        Is final if agent is on final position.
        :param state:
        :return:
        """
        return state[0] in self.finals.keys()

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment train_data
        del data['pits']
        del data['bonus']

        return data

    def transition_reward(self, state: tuple, action: int, next_state: tuple) -> Vector:

        # Separate position from bonus_activated
        position, bonus_activated = next_state

        # Default reward
        reward = self.default_reward.copy()

        # Get treasure value
        reward[0], reward[1] = self.finals.get(position, (reward[0], reward[1]))

        # If the bonus is activated, double the reward.
        if bonus_activated:
            reward[0] *= 2
            reward[1] *= 2

        return reward

    def states(self) -> set:

        # Unpack spaces
        position, bonus_activate = self.observation_space.spaces
        x_position, y_position = position.spaces

        # Get all positions
        all_positions = {(x, y) for x in range(x_position.n) for y in range(y_position.n)}

        # Get obstacles, finals positions and pits
        finals_obstacles_and_pits = self.obstacles.union(set(self.finals.keys())).union(self.pits)

        # Generate available states
        available_states = set(product(all_positions - finals_obstacles_and_pits, {True, False}))

        # Remove impossible states
        available_states = available_states - {
            ((3, 3), False)
        }

        # Return all available spaces
        return available_states
