"""
The agent controls a spaceship which starts each episode in the location marked ’S’ and aims to discover a
habitable planet while minimising the amount of radiation to which it is exposed. A penalty of −1 is received for the
radiation objective on all time-steps, except when in a region of high radiation (marked ’R’) when the penalty is
−11. A positive reward is received for the mission success objective whenever a terminal position corresponding to a
planet is reached – the magnitude of this reward reflects the desirability of that planet. If the ship enters a cell
occupied by an asteroid, the ship is destroyed, the episode ends, and the agent receives a mission success reward of
−100. The threshold is applied to the mission success objective, meaning that the agent will attempt to minimise
radiation exposure subject to meeting minimum habitability requirements. in Space Exploration the agent can move to
all eight neighbouring states (i.e. there are eight actions). Also if the agent leaves the bounds of the grid,
it moves to the opposite edge of the grid. For example if the agent moves up from the top row of the grid,
it will move to the bottom row of the same column).

FINAL STATES: To reach any of x-value states.

HV REFERENCE: (-100, -150)

REF: P. Vamplew et al. (2017)
"""
import gym

import utils.environments as ue
from models import Vector
from .env_mesh import EnvMesh


class SpaceExploration(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'UP RIGHT': 1, 'RIGHT': 2, 'DOWN RIGHT': 3, 'DOWN': 4, 'DOWN LEFT': 5, 'LEFT': 6, 'UP LEFT': 7}

    # Experiments common hypervolume reference
    hv_reference = Vector([-100, -150])

    def __init__(self, initial_state: tuple = (5, 2), default_reward: tuple = (0, -1), seed: int = 0,
                 action_space: gym.spaces = None):
        """
        :param initial_state: Initial state where start the agent.
        :param default_reward: (mission_success, radiation)
        :param seed: Seed used for np.random.RandomState method.
        """

        # List of all treasures and its reward.
        finals = {}
        finals.update({(0, i): 20 for i in range(5)})
        finals.update({(9, i): 10 for i in range(3)})
        finals.update({(12, i): 30 for i in range(5)})

        obstacles = frozenset()
        mesh_shape = (13, 5)
        default_reward = Vector(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, initial_state=initial_state, default_reward=default_reward,
                         finals=finals, obstacles=obstacles, action_space=action_space)

        self.asteroids = {
            (5, 0), (4, 1), (6, 1), (3, 2), (7, 2), (4, 3), (6, 3), (5, 4)
        }

        # Define radiations states (If the agent is on any of these, then receive -100 penalization)
        self.radiations = set()
        self.radiations = self.radiations.union({(1, i) for i in range(5)})
        self.radiations = self.radiations.union({(10, i) for i in range(5)})
        self.radiations = self.radiations.union({(11, i) for i in range(5)})

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (position, (mission_success, radiation), final, extra)
        """

        # Initialize reward as vector
        reward = self.default_reward.copy()

        # Update previous state
        self.current_state = self.next_state(action=action)

        # If the ship crash with asteroid, the ship is destroyed. else mission success.
        reward[0] = -100 if self.current_state in self.asteroids else self.finals.get(
            self.current_state, self.default_reward[0]
        )

        # If agent is in a radiation position, the penalty is -11, else is default radiation
        reward[1] = -11 if self.current_state in self.radiations else self.default_reward[1]

        # Check if is_final
        final = self.is_final(self.current_state)

        # Set extra
        info = {}

        return self.current_state, reward, final, info

    def next_position(self, action: int, position: tuple) -> (tuple, bool):
        """
        Given an action and a position, return the next position reached.
        :param action:
        :param position:
        :return:
        """

        # Get my position
        x, y = position

        # Get observations spaces
        observation_space_x, observation_space_y = self.observation_space.spaces

        # Do movement in cyclic mesh
        if action == self.actions['UP']:
            y = ue.move_up(y=y, limit=observation_space_y.n)
        elif action == self.actions['RIGHT']:
            x = ue.move_right(x=x, limit=observation_space_x.n)
        elif action == self.actions['DOWN']:
            y = ue.move_down(y=y, limit=observation_space_y.n)
        elif action == self.actions['LEFT']:
            x = ue.move_left(x=x, limit=observation_space_x.n)
        elif action == self.actions['UP RIGHT']:
            y = ue.move_up(y=y, limit=observation_space_y.n)
            x = ue.move_right(x=x, limit=observation_space_x.n)
        elif action == self.actions['DOWN RIGHT']:
            y = ue.move_down(y=y, limit=observation_space_y.n)
            x = ue.move_right(x=x, limit=observation_space_x.n)
        elif action == self.actions['DOWN LEFT']:
            y = ue.move_down(y=y, limit=observation_space_y.n)
            x = ue.move_left(x=x, limit=observation_space_x.n)
        elif action == self.actions['UP LEFT']:
            y = ue.move_up(y=y, limit=observation_space_y.n)
            x = ue.move_left(x=x, limit=observation_space_x.n)

        # Set next position
        next_position = x, y

        return next_position, True

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next position with current position and action given, in this environment is 8-neighbors.
        :param state: If a position is given, do action from that position.
        :param action: from action_space
        :return:
        """

        # Get my position
        position = state if state else self.current_state

        next_position, is_valid = self.next_position(action=action, position=position)

        if not self.observation_space.contains(next_position) or not is_valid:
            next_position = position

        # Return (x, y) position
        return next_position

    def is_final(self, state: tuple = None) -> bool:
        """
        Is final if agent crash with asteroid or is on final position.
        :param state:
        :return:
        """

        # Check if agent crash with asteroid
        crash = state in self.asteroids

        # Check if agent is in final position
        final = state in self.finals.keys()

        return crash or final

    def transition_reward(self, state: tuple, action: int, next_state: tuple) -> Vector:
        """
        Return reward for reach `next_state` from `state` using `action`.

        :param state: initial position
        :param action: action to do
        :param next_state: next position reached
        :return:
        """
        # Initialize reward as vector
        reward = self.default_reward.copy()

        # If the ship crash with asteroid, the ship is destroyed. else mission success.
        reward[0] = -100 if next_state in self.asteroids else self.finals.get(
            next_state, reward[0]
        )

        # If agent is in a radiation position, the penalty is -11, else is default radiation
        reward[1] = -11 if next_state in self.radiations else reward[1]

        return reward
