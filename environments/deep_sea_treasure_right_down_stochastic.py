"""
This is a variant of the DeepSeaTreasureRightDown environment, where allowed
actions (only DOWN_PROB or RIGHT_PROB) are stochastic. In the last column (i.e.
maximum value of x), only a deterministic DOWN action is allowed. The transition
probability model is provided in the __init__ method.

HV REFERENCE: (-25, 0)
"""

import utils.environments as ue
from environments import DeepSeaTreasureRightDown
from spaces import Bag


class DeepSeaTreasureRightDownStochastic(DeepSeaTreasureRightDown):
    # Possible actions
    _actions = {'RIGHT_PROB': 0, 'DOWN_PROB': 1, 'DOWN': 2}

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0,), seed: int = 0,
                 p_stochastic: float = 0.8, columns: int = 10):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        :param columns: Number of diagonals to use with this environment.
        :param p_stochastic: transition probability model. Is the probability of achieving the desired result of the
        action (i.e. moving right in RIGHT_PROB, and down with DOWN_PROB).
        """

        super().__init__(initial_state=initial_state, default_reward=default_reward, seed=seed, columns=columns)

        # Prepare stochastic p_stochastic
        self.p_stochastic = p_stochastic

    def next_position(self, action: int, position: tuple) -> (tuple, bool):
        # Get my position
        x, y = position

        # Do movement
        if action == self.actions['RIGHT_PROB']:
            rnd_number = self.np_random.uniform()

            if self.p_stochastic > rnd_number:
                x += 1
            else:
                y += 1

        elif action == self.actions['DOWN_PROB']:
            rnd_number = self.np_random.uniform()

            if self.p_stochastic > rnd_number:
                y += 1
            else:
                x += 1
        elif action == self.actions['DOWN']:
            y += 1

        # Set next position
        next_position = x, y

        # If obstacles exist, check if next_position is in self.obstacles
        is_obstacle = bool(self.obstacles) and next_position in self.obstacles

        return next_position, not is_obstacle

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

        # Can we go to right?
        x_right = x + 1

        # Check if we are in a border of mesh
        if x_right < self.observation_space[0].n:
            possible_actions.append(self.actions['RIGHT_PROB'])
            possible_actions.append(self.actions['DOWN_PROB'])
        else:
            possible_actions.append(self.actions['DOWN'])

        # Setting to dynamic_space
        self._action_space.items = possible_actions

        # Update n length
        self._action_space.n = len(possible_actions)

        # Return a list of iterable valid actions
        return self._action_space

    def transition_probability(self, state: tuple, action: int, next_state: tuple) -> float:

        probability = 1.

        if action == self.actions['DOWN_PROB']:

            # Next position is on right
            if ue.is_on_right_or_same_position(state=state, next_state=next_state):
                probability = self.p_stochastic
            # Next position is on down
            elif ue.is_on_down_or_same_position(state=state, next_state=next_state):
                probability = 1. - self.p_stochastic

        elif action == self.actions['RIGHT_PROB']:

            # Next position is on right
            if ue.is_on_right_or_same_position(state=state, next_state=next_state):
                probability = 1. - self.p_stochastic
            # Next position is on down
            elif ue.is_on_down_or_same_position(state=state, next_state=next_state):
                probability = self.p_stochastic

        return probability

    def reachable_states(self, state: tuple, action: int) -> set:

        # Unpack position
        x, y = state

        # Reachable states from that position with that action
        reachable_states = set()

        if x >= (self.observation_space[0].n - 1) and (
                action == self.actions['RIGHT_PROB'] or action == self.actions['DOWN_PROB']
        ):
            raise ValueError('Action/State combination returns a cyclic position.')

        if action == self.actions['DOWN']:
            reachable_states.add((x, y + 1))
        elif action == self.actions['DOWN_PROB'] or action == self.actions['RIGHT_PROB']:
            # Go to down
            reachable_states.add((x, y + 1))
            # Go to right
            reachable_states.add((x + 1, y))

        return reachable_states
