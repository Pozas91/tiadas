"""
This is a variant of original problem of DeepSeaTreasure where we only one or
two actions are allowed in each position. For states in the rightmost column
(i.e. the largest possible second component) only DOWN is allowed. For all other
states, RIGHT and DOWN are allowed.

Notice that the constructor allows a 'diagonals' parameter that can be used to indicate the
number of diagonals in the environment to be considered, starting from the left
hand side. This allows experimenting with 'subspaces' in the domain, i.e.
the same environment, buy considering only the first k diagonals.

Notice that is_final does not consider here a maximum number of steps for each
episode (while DeepSeaTreasure does).

All other elements of the environment behave are as in DeepSeaTreasure.

HV REFERENCE: (-25, 0)
"""
from colorama import Fore, init

from spaces import Bag
from .deep_sea_treasure import DeepSeaTreasure

init(autoreset=True)


class DeepSeaTreasureRightDown(DeepSeaTreasure):
    # Possible actions
    _actions = {'RIGHT': 0, 'DOWN': 1}

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0,), seed: int = 0, columns: int = 10):
        """
        :param initial_state: Initial state where start the agent.
        :param default_reward: (time_inverted, treasure_value)
        :param seed: Seed used for np.random.RandomState method.
        :param columns: Number of columns to be used to build this environment (allows experimenting with an identical
                        environment, but considering only the first k columns) (By default 10).
        """

        # Action space
        action_space = Bag([])
        action_space.seed(seed=seed)

        super().__init__(initial_state=initial_state, default_reward=default_reward, seed=seed, columns=columns,
                         action_space=action_space)

    def next_position(self, action: int, position: tuple) -> (tuple, bool):
        """
        Given a position and an action, returns the next_position
        :param action:
        :param position:
        :return:
        """
        # Unpack
        x, y = position

        # Do movement
        if action == self.actions['RIGHT']:
            x += 1
        elif action == self.actions['DOWN']:
            y += 1

        # Set next position
        next_position = x, y

        # If exists obstacles, then next_position must be in self.obstacles
        is_obstacle = bool(self.obstacles) and next_position in self.obstacles

        return next_position, not is_obstacle

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next position with current position and action given. Default is 4-neighbors (UP, LEFT, DOWN, RIGHT)
        :param state: If a position is given, do action from that position.
        :param action: from action_space
        :return: a new position (or old if is invalid action)
        """

        # Get my position
        position = state if state else self.current_state

        next_position, is_valid = self.next_position(action=action, position=position)

        if not self.observation_space.contains(next_position) or not is_valid or position == next_position:
            raise ValueError('Action/State combination returns a cyclic position.')

        # Return (x, y) position
        return next_position

    @property
    def action_space(self) -> Bag:
        """
        Get a dynamic action space with only valid actions.
        Only DOWN is possible if the submarine reached the last column
        (i.e. the largest possible x value). Otherwise, both DOWN and 
        RIGHT are available.
        :return:
        """

        # Get current position
        x, y = self.current_state

        # Setting possible actions
        possible_actions = []

        # Can we go to right?
        x_right = x + 1

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x_right, y)):
            # We can go to right
            possible_actions.append(self.actions['RIGHT'])

        # We always can go to down (Because an final position stop the problem)
        possible_actions.append(self.actions['DOWN'])

        # Setting to dynamic_space
        self._action_space.items = possible_actions

        # Update n length
        self._action_space.n = len(possible_actions)

        # Return a list of iterable valid actions
        return self._action_space
