"""
Inspired by the Deep Sea Treasure (DST) environment. In contrast to the, the values of the treasures are altered to
create a convex Pareto front.

FINAL STATE: To reach any final position.

REF: Multi-objective reinforcement learning using sets of pareto dominating policies (Kristof Van Moffaert,
Ann Nowé) 2014

HV REFERENCE: (-25, 0, -120)
"""

from environments import PressurizedBountifulSeaTreasure
from spaces import Bag
import utils.environments as ue


class PressurizedBountifulSeaTreasureRightDownStochastic(PressurizedBountifulSeaTreasure):
    # Possible actions
    _actions = {'RIGHT_PROB': 0, 'DOWN_PROB': 1, 'DOWN': 2}

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0,), seed: int = 0, columns: int = 10,
                 p_stochastic: float = 0.8):
        """
        :param initial_state:
        :param default_reward: (treasure_value)
        :param seed:
        :param columns:
        :param p_stochastic:
        """

        action_space = Bag([])
        action_space.seed(seed=seed)

        super().__init__(initial_state=initial_state, default_reward=default_reward, seed=seed, columns=columns,
                         action_space=action_space)

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

        # If exists obstacles, then next_position must be in self.obstacles
        is_obstacle = bool(self.obstacles) and next_position in self.obstacles

        return next_position, not is_obstacle

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next position with current position and action given.
        :param state: If a position is given, do action from that position.
        :param action: from action_space
        :return: a new position (or old if is invalid action)
        """

        # Get my position
        position = state if state else self.current_state

        next_position, is_valid = self.next_position(action=action, position=position)

        if not self.observation_space.contains(next_position) or not is_valid or position == next_position:
            raise ValueError("Action/State combination isn't valid.")

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
