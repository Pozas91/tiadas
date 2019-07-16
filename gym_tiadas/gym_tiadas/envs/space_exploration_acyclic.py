"""
A variant of Space Exploration environment for acyclic agents.

Remove all rewards from left side, start at (0, 0) state, and remove some asteroids states. We can go only to RIGHT,
DOWN RIGHT and DOWN
"""
from models import Vector
from spaces import DynamicSpace
from .env_mesh import EnvMesh


class SpaceExplorationAcyclic(EnvMesh):
    # Possible actions
    _actions = {'RIGHT': 0, 'DOWN RIGHT': 1, 'DOWN': 2}

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0, -1), seed: int = 0):
        """
        :param initial_state:
        :param default_reward: (mission_success, radiation)
        :param seed:
        """

        # List of all treasures and its reward.
        finals = {}
        finals.update({(9, i): 20 for i in range(3)})
        finals.update({(12, i): 30 for i in range(5)})

        obstacles = frozenset()
        mesh_shape = (13, 5)
        default_reward = Vector(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, initial_state=initial_state, default_reward=default_reward,
                         finals=finals, obstacles=obstacles)

        # Define asteroids states
        self.asteroids = {
            # (3, 2),
            # (4, 1), (4, 3),
            (5, 0), (5, 4),
            (6, 3), (6, 1),
            (7, 2)
        }

        # Define radiation states
        self.radiations = set()
        self.radiations = self.radiations.union({(10, i) for i in range(5)})
        self.radiations = self.radiations.union({(11, i) for i in range(5)})

        # Trying improve performance
        self.dynamic_action_space = DynamicSpace([])
        self.dynamic_action_space.seed(seed=seed)

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
        if action == self._actions['RIGHT']:
            x = self.__move_right(x=x, limit=observation_space_x.n)
        elif action == self._actions['DOWN']:
            y = self.__move_down(y=y, limit=observation_space_y.n)
        elif action == self._actions['DOWN RIGHT']:
            y = self.__move_down(y=y, limit=observation_space_y.n)
            x = self.__move_right(x=x, limit=observation_space_x.n)

        # Set new state
        new_state = x, y

        if not self.observation_space.contains(new_state):
            raise ValueError('Invalid action, that action produces cycles.')

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

    @property
    def action_space(self) -> DynamicSpace:
        """
        Get a dynamic action space with only valid actions.
        :return:
        """

        # Get current state
        x, y = self.current_state

        # Get observations spaces
        observation_space_x, observation_space_y = self.observation_space.spaces

        # Setting possible actions
        possible_actions = []

        # Can we go to RIGHT?
        x_right = self.__move_right(x=x, limit=observation_space_x.n)

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x_right, y)):
            # We can go to right
            possible_actions.append(self._actions['RIGHT'])

        # Can we go to DOWN RIGHT?
        x_right = self.__move_right(x=x, limit=observation_space_x.n)
        y_down = self.__move_down(y=y, limit=observation_space_y.n)

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x_right, y_down)):
            # We can go to right
            possible_actions.append(self._actions['DOWN RIGHT'])

        # Can we go to DOWN?
        y_down = y + 1

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x, y_down)):
            # We can go to right
            possible_actions.append(self._actions['DOWN'])

        # Setting to dynamic_space
        self.dynamic_action_space.items = possible_actions

        # Update n length
        self.dynamic_action_space.n = len(possible_actions)

        # Return a list of iterable valid actions
        return self.dynamic_action_space

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment data
        del data['radiations']
        del data['asteroids']
        del data['dynamic_action_space']

        return data
