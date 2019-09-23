"""
This is a variant of original problem of DeepSeaTreasure where we only one or
two actions are allowed in each state. For states in the rightmost column
(i.e. the largest possible second component) only DOWN is allowed. For all other
states, RIGHT and DOWN are allowed.

Notice that the constructor allows a 'columns' parameter that can be used to indicate the
number of columns in the environment to be considered, starting from the left
hand side. This allows experimenting with 'subspaces' in the domain, i.e.
the same environment, buy considering only the first k columns.

Notice that is_final does not consider here a maximum number of steps for each
episode (while DeepSeaTreasure does).

All other elements of the environment behave are as in DeepSeaTreasure.

The reference point for hypervolume calculations in this environment is (-25, 0).
"""
from models import Vector
from spaces import DynamicSpace
from .env_mesh import EnvMesh


class DeepSeaTreasureRightDown(EnvMesh):
    # Possible actions
    _actions = {'RIGHT': 0, 'DOWN': 1}

    # Pareto optimal policy vector-values.
    pareto_optimal = [
        (-1, 1), (-3, 2), (-5, 3), (-7, 5), (-8, 8), (-9, 16), (-13, 24), 
        (-14, 50), (-17, 74), (-19, 124)
    ]

    def __init__(self, initial_state: tuple = (0, 0), 
                 default_reward: tuple = (0,), 
                 seed: int = 0, 
                 columns: int = 0):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        :param columns: Number of columns to be used to build this environment
                        (allows experimenting with an identical environment,
                        but considering only the first k columns).
        """

        #the original full-size environment.
        original_mesh_shape = (10, 11)

        if columns < 1 or columns > original_mesh_shape[0]:
            columns = original_mesh_shape[0]

        # Dictionary with final states as keys, and treasure amounts as value.
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

        finals = dict(filter(lambda x: x[0][0] < columns, finals.items()))

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
        obstacles = frozenset(filter(lambda x: x[0] < columns, obstacles))

        #subspace of the environment to be considered
        mesh_shape = (columns, 11)

        # Default reward plus time (time_inverted, treasure_value)
        default_reward = (-1,) + default_reward
        default_reward = Vector(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, initial_state=initial_state, default_reward=default_reward,
                         finals=finals, obstacles=obstacles)

        # Trying improve performance
        self.dynamic_action_space = DynamicSpace([])
        self.dynamic_action_space.seed(seed=seed)

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step in the environment
        :param action:
        :return: (state, (steps_taken, treasure_value), final, info)
        """

        # Initialize rewards as vector
        rewards = self.default_reward.copy()

        # Get new state
        new_state = self.next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Get treasure value
        rewards[1] = self.finals.get(self.current_state, self.default_reward[1])

        # Set info
        info = {}

        # Check is_final
        final = self.is_final(self.current_state)

        return self.current_state, rewards, final, info

    def reset(self) -> tuple:
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        return self.current_state

    def is_final(self, state: tuple = None) -> bool:
        """
        Return True if state given is final, False in otherwise.
        :param state:
        :return:
        """
        return state in self.finals.keys()

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next state with current state and action given. Default is 4-neighbors (UP, LEFT, DOWN, RIGHT)
        :param state: If a state is given, do action from that state.
        :param action: from action_space
        :return: a new state (or old if is invalid action)
        """

        # Get my position
        x, y = state if state else self.current_state

        # Do movement
        if action == self._actions['RIGHT']:
            x += 1
        elif action == self._actions['DOWN']:
            y += 1

        # Set new state
        new_state = x, y

        # If exists obstacles, then new_state must be in self.obstacles
        is_obstacle = bool(self.obstacles) and new_state in self.obstacles

        if not self.observation_space.contains(new_state) or is_obstacle or state == new_state:
            raise ValueError("Action/State combination isn't valid.")

        # Return (x, y) position
        return new_state

    @property
    def action_space(self) -> DynamicSpace:
        """
        Get a dynamic action space with only valid actions.
        Only DOWN is possible if the submarine reached the last column
        (i.e. the largest possible x value). Otherwise, both DOWN and 
        RIGHT are available.
        :return:
        """

        # Get current state
        x, y = self.current_state

        # Setting possible actions
        possible_actions = []

        # Can we go to right?
        x_right = x + 1

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x_right, y)):
            # We can go to right
            possible_actions.append(self._actions['RIGHT'])

        # We always can go to down (Because an final state stop the problem)
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
        del data['dynamic_action_space']

        return data
