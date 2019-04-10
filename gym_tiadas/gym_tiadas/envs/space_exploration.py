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

REF: P. Vamplew et al. (2017)
"""

from .env_mesh import EnvMesh


class SpaceExploration(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'UP RIGHT': 1, 'RIGHT': 2, 'DOWN RIGHT': 3, 'DOWN': 4, 'DOWN LEFT': 5, 'LEFT': 6, 'UP LEFT': 7}

    def __init__(self, mesh_shape=(13, 5), initial_state=(5, 2), default_reward=0., seed=0):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        """

        # List of all treasures and its reward.
        finals = {}
        finals.update({(0, i): 20 for i in range(5)})
        finals.update({(9, i): 10 for i in range(3)})
        finals.update({(12, i): 30 for i in range(5)})

        obstacles = dict()

        super().__init__(mesh_shape, seed, initial_state=initial_state, default_reward=default_reward, finals=finals,
                         obstacles=obstacles)

        self.asteroids = {
            (5, 0), (4, 1), (6, 1), (3, 2), (7, 2), (4, 3), (6, 3), (5, 4)
        }

        self.radiations = set()
        self.radiations = self.radiations.union({(1, i) for i in range(5)})
        self.radiations = self.radiations.union({(10, i) for i in range(5)})
        self.radiations = self.radiations.union({(11, i) for i in range(5)})

    def step(self, action) -> (object, [float, float], bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (mission_success, radiation), final, info)
        """

        # (mission_success, radiation)
        rewards = [0, 0]

        # Get new state
        new_state = self._next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # If agent is in a radiation state, the penalty is -11, else is -1.
        rewards[1] = -11 if self.current_state in self.radiations else -1

        # If the ship crash with asteroid, the ship is destroyed. else mission success.
        rewards[0] = -100 if self.current_state in self.asteroids else self.finals.get(self.current_state,
                                                                                       self.default_reward)

        # Check if is_final
        final = self.is_final(self.current_state)

        # Set info
        info = {}

        return self.current_state, rewards, final, info

    def reset(self):
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        return self.current_state

    def _next_state(self, action) -> object:
        """
        Calc next state with current state and action given, in this environment is 8-neighbors.
        :param action:
        :return:
        """

        # Get my position
        x, y = self.current_state
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
    def __move_up(y, limit=5):
        """
        Move to up
        :param y:
        :param limit:
        :return:
        """
        return (y if y > 0 else limit) - 1

    @staticmethod
    def __move_right(x, limit=13):
        """
        Move to right
        :param x:
        :param limit:
        :return:
        """
        return (x + 1) % limit

    @staticmethod
    def __move_down(y, limit=5):
        """
        Move to down
        :param y:
        :param limit:
        :return:
        """
        return (y + 1) % limit

    @staticmethod
    def __move_left(x, limit=13):
        """
        Move to left
        :param x:
        :param limit:
        :return:
        """
        return (x if x > 0 else limit) - 1

    def is_final(self, state=None) -> bool:
        # Check if agent crash with asteroid
        crash = state in self.asteroids
        # Check if agent is in final state
        final = state in self.finals.keys()

        return crash or final
