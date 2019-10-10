"""The agent controls a spaceship which starts each episode in the location marked ’S’ and aims to discover a
habitable planet while minimising the amount of radiation to which it is exposed. A penalty of −1 is received for the
radiation objective on all time-steps, except when in a region of high radiation (marked ’R’) when the penalty is
−11. A positive reward is received for the mission success objective whenever a terminal state corresponding to a
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
from models import Vector
from .env_mesh import EnvMesh


class SpaceExploration(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'UP RIGHT': 1, 'RIGHT': 2, 'DOWN RIGHT': 3, 'DOWN': 4, 'DOWN LEFT': 5, 'LEFT': 6, 'UP LEFT': 7}

    # Experiments common hypervolume reference
    hv_reference = Vector([-100, -150])

    def __init__(self, initial_state: tuple = (5, 2), default_reward: tuple = (0, -1), seed: int = 0):
        """
        :param initial_state:
        :param default_reward: (mission_success, radiation)
        :param seed:
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
                         finals=finals, obstacles=obstacles)

        self.asteroids = {
            (5, 0), (4, 1), (6, 1), (3, 2), (7, 2), (4, 3), (6, 3), (5, 4)
        }

        self.radiations = set()
        self.radiations = self.radiations.union({(1, i) for i in range(5)})
        self.radiations = self.radiations.union({(10, i) for i in range(5)})
        self.radiations = self.radiations.union({(11, i) for i in range(5)})

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (mission_success, radiation), final, info)
        """

        # Initialize rewards as vector
        rewards = self.default_reward.copy()

        # Get new state
        new_state = self.next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # If the ship crash with asteroid, the ship is destroyed. else mission success.
        rewards[0] = -100 if self.current_state in self.asteroids else self.finals.get(self.current_state,
                                                                                       self.default_reward[0])

        # If agent is in a radiation state, the penalty is -11, else is default radiation
        rewards[1] = -11 if self.current_state in self.radiations else self.default_reward[1]

        # Check if is_final
        final = self.is_final(self.current_state)

        # Set info
        info = {}

        return self.current_state, rewards, final, info

    def reset(self) -> tuple:
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        return self.current_state

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next state with current state and action given, in this environment is 8-neighbors.
        :param state: If a state is given, do action from that state.
        :param action: from action_space
        :return:
        """

        # Get my position
        x, y = state if state else self.current_state

        # Get observations spaces
        observation_space_x, observation_space_y = self.observation_space.spaces

        # Do movement in cyclic mesh
        if action == self._actions.get('UP'):
            y = self.__move_up(y=y, limit=observation_space_y.n)
        elif action == self._actions.get('RIGHT'):
            x = self.__move_right(x=x, limit=observation_space_x.n)
        elif action == self._actions.get('DOWN'):
            y = self.__move_down(y=y, limit=observation_space_y.n)
        elif action == self._actions.get('LEFT'):
            x = self.__move_left(x=x, limit=observation_space_x.n)
        elif action == self._actions.get('UP RIGHT'):
            y = self.__move_up(y=y, limit=observation_space_y.n)
            x = self.__move_right(x=x, limit=observation_space_x.n)
        elif action == self._actions.get('DOWN RIGHT'):
            y = self.__move_down(y=y, limit=observation_space_y.n)
            x = self.__move_right(x=x, limit=observation_space_x.n)
        elif action == self._actions.get('DOWN LEFT'):
            y = self.__move_down(y=y, limit=observation_space_y.n)
            x = self.__move_left(x=x, limit=observation_space_x.n)
        elif action == self._actions.get('UP LEFT'):
            y = self.__move_up(y=y, limit=observation_space_y.n)
            x = self.__move_left(x=x, limit=observation_space_x.n)

        # Set new state
        new_state = x, y

        if not self.observation_space.contains(new_state):
            # New state is invalid.
            new_state = self.current_state

        # Return (x, y) position
        return new_state

    @staticmethod
    def __move_up(y: int, limit: int = 5) -> int:
        """
        Move to up
        :param y:
        :param limit:
        :return:
        """
        return (y if y > 0 else limit) - 1

    @staticmethod
    def __move_right(x: int, limit: int = 13) -> int:
        """
        Move to right
        :param x:
        :param limit:
        :return:
        """
        return (x + 1) % limit

    @staticmethod
    def __move_down(y: int, limit: int = 5) -> int:
        """
        Move to down
        :param y:
        :param limit:
        :return:
        """
        return (y + 1) % limit

    @staticmethod
    def __move_left(x: int, limit: int = 13) -> int:
        """
        Move to left
        :param x:
        :param limit:
        :return:
        """
        return (x if x > 0 else limit) - 1

    def is_final(self, state: tuple = None) -> bool:
        """
        Is final if agent crash with asteroid or is on final state.
        :param state:
        :return:
        """

        # Check if agent crash with asteroid
        crash = state in self.asteroids
        # Check if agent is in final state
        final = state in self.finals.keys()

        return crash or final

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment data
        del data['radiations']
        del data['asteroids']

        return data
