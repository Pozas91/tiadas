# coding=utf-8
"""
Variant of BonusWorld environment to acyclic agents. If agent there on pit position, episode ends and agent receives
(-50, -50) reward.

HV REFERENCE: (-50, -50, -50)
"""

from environments import BonusWorld
from models import Vector
from spaces import Bag


class BonusWorldAcyclic(BonusWorld):
    # Possible actions
    _actions = {'RIGHT': 0, 'DOWN': 1}

    def __init__(self, initial_state: tuple = ((0, 0), False), default_reward: tuple = (0, 0), seed: int = 0):
        """
        :param initial_state: Initial state where start the agent.
        :param default_reward: (objective 1, objective 2)
        :param seed: Seed used for np.random.RandomState method.
        """

        # Create a bag action space
        action_space = Bag([])
        action_space.seed(seed)

        super().__init__(seed=seed, initial_state=initial_state, default_reward=default_reward,
                         action_space=action_space)

        # Set obstacles
        self.obstacles = frozenset({
            (2, 2)
        })

        # PITS are finals states in this variant
        self.finals.update({
            state: Vector([-50, -50]) for state in self.pits
        })

        self.pits = list()

    def next_position(self, action: int, position: tuple) -> (tuple, bool):
        """
        Given a position and an action, return the next position of this environment
        :param action:
        :param position:
        :return:
        """
        # Unpack position (x, y)
        x, y = position

        # Do movement
        if action == self.actions['RIGHT']:
            x += 1
        elif action == self.actions['DOWN']:
            y += 1

        # Set next position
        next_position = x, y

        # If exists obstacles, then new_position must be in self.obstacles
        is_obstacle = bool(self.obstacles) and next_position in self.obstacles

        return next_position, not is_obstacle

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next position with current position and action given. Default is 2-neighbors (DOWN, RIGHT)
        :param state: If a position is given, do action from that position.
        :param action: from action_space
        :return: a new position (or old if is invalid action)
        """

        # Unpack complex position (position, bonus_activated)
        position, bonus_activated = state if state else self.current_state

        next_position, is_valid = self.next_position(action=action, position=position)

        if not self.observation_space[0].contains(next_position) or not is_valid or position == next_position:
            raise ValueError('Action/State combination returns a cyclic position.')

        if next_position in self.bonus:
            bonus_activated = True

        # Build next position with the new position
        return next_position, bonus_activated

    @property
    def action_space(self) -> Bag:
        """
        Get a dynamic action space with only valid actions.
        :return:
        """

        # Get current position
        position, bonus = self.current_state

        # Unpack position
        x, y = position

        # Setting possible actions
        possible_actions = []

        # Can we go to right?
        x_right = x + 1

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains(((x_right, y), bonus)) and (x_right, y) not in self.obstacles:
            # We can go to right
            possible_actions.append(self.actions['RIGHT'])

        # Can we go to down?
        y_down = y + 1

        # Check that y_down is not and obstacle and is into mesh
        if self.observation_space.contains(((x, y_down), bonus)) and (x, y_down) not in self.obstacles:
            # We can go to down
            possible_actions.append(self.actions['DOWN'])

        # Setting to dynamic_space
        self._action_space.items = possible_actions

        # Update n length
        self._action_space.n = len(possible_actions)

        # Return a list of iterable valid actions
        return self._action_space
