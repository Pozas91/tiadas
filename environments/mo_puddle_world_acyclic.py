"""
Variant of Mo Puddle World for Acyclic agents.

HV REFERENCE: (-50, -150)
"""
from colorama import Fore
from scipy.spatial import distance

from environments import MoPuddleWorld
from models import Vector
from spaces import Bag


class MoPuddleWorldAcyclic(MoPuddleWorld):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1}

    def __init__(self, default_reward: tuple = (10, 0), penalize_non_goal: float = -1, seed: int = 0,
                 final_state: tuple = (19, 0)):
        """
        :param default_reward: (non_goal_reached, puddle_penalize)
        :param penalize_non_goal: While agent does not reach a final position get a penalize.
        :param seed:
        :param final_state: This environment only has a final position.
        """

        action_space = Bag([])
        action_space.seed(seed=seed)

        super().__init__(default_reward=default_reward, penalize_non_goal=penalize_non_goal, seed=seed,
                         final_state=final_state, action_space=action_space)

    def next_position(self, action: int, position: tuple) -> (tuple, bool):
        # Unpack position (x, y)
        x, y = position

        # Do movement
        if action == self.actions['RIGHT']:
            x += 1
        elif action == self.actions['UP']:
            y -= 1

        # Set next position
        next_position = x, y

        return next_position, True

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next position with current position and action given. Default is 2-neighbors (UP, RIGHT)
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
        :return:
        """

        # Get current position
        x, y = self.current_state

        # Setting possible actions
        possible_actions = []

        # Can we go to RIGHT?
        x_right = x + 1

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x_right, y)):
            # We can go to right
            possible_actions.append(self.actions['RIGHT'])

        # Can we go to UP?
        y_up = y - 1

        # Check that y_down is not and obstacle and is into mesh
        if self.observation_space.contains((x, y_up)):
            # We can go to down
            possible_actions.append(self.actions['UP'])

        # Setting to dynamic_space
        self._action_space.items = possible_actions

        # Update n length
        self._action_space.n = len(possible_actions)

        # Return a list of iterable valid actions
        return self._action_space

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

    def calc_puddle_penalization(self, state: tuple):

        # Get free spaces
        free_spaces = set(filter(lambda x: x[0] >= state[0] and x[1] <= state[1], self.free_spaces))

        # Min distance found!
        min_distance = min(distance.cityblock(x, state) for x in free_spaces)

        # Set penalization per distance
        return -min_distance
