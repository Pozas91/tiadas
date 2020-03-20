"""
Such as DeepSeaTreasure environment but has a vector of p_stochastic probabilities, which will be used when an action
is to be taken.

HV REFERENCE: (-25, 0)
"""
import utils.environments as ue
from environments import DeepSeaTreasure
from models import Vector


class DeepSeaTreasureStochastic(DeepSeaTreasure):

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0,), seed: int = 0,
                 n_transition: float = 0.3, columns: int = 10):
        """
        :param initial_state: start position for each episode.
        :param default_reward: (time_inverted, treasure_value)
        :param seed: Initial initial_seed for the random number generator.
        :param n_transition: if is 0, the action affected by the transition is always the same action.
        """

        super().__init__(initial_state=initial_state, default_reward=default_reward, seed=seed, columns=columns)

        # Transaction
        assert 0 <= n_transition <= 1.

        # [DIR_0, DIR_90, DIR_180, DIR_270] transaction tuple
        self.transitions = (1. - n_transition, n_transition / 3, n_transition / 3, n_transition / 3)

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (position, (time_inverted, treasure_value), final, extra)
        """

        # Get probability action
        action = self.__probability_action(action=action)

        return super().step(action=action)

    def __probability_action(self, action: int) -> int:
        """
        Decide probability action after apply probabilistic p_stochastic.
        :param action:
        :return:
        """

        # Get a random uniform number [0., 1.]
        random = self.np_random.uniform()

        # Start with first direction
        direction = 0

        # Accumulate roulette
        roulette = self.transitions[direction]

        # While random is greater than roulette
        while random > roulette:
            # Increment action
            direction += 1

            # Increment roulette
            roulette += self.transitions[direction]

        # Cyclic direction
        return (direction + action) % self.action_space.n

    def transition_probability(self, state: tuple, action: int, next_state: tuple) -> float:
        """
        Return probability to reach `next_state` from `position` using `action`.

        In non-stochastic environments this return always 1.

        :param state: initial position
        :param action: action to do
        :param next_state: next position reached
        :return:
        """

        probability = self.transitions[1]

        straight_movement = (
                (action == self.actions['UP'] and ue.is_on_up_or_same_position(
                    state=state, next_state=next_state)
                 ) or
                (action == self.actions['RIGHT'] and ue.is_on_right_or_same_position(
                    state=state, next_state=next_state)
                 ) or
                (action == self.actions['DOWN'] and ue.is_on_down_or_same_position(
                    state=state, next_state=next_state)
                 ) or
                (action == self.actions['LEFT'] and ue.is_on_left_or_same_position(
                    state=state, next_state=next_state
                ))
        )

        if straight_movement:
            probability = self.transitions[0]

        return probability

    def reachable_states(self, state: tuple, action: int) -> set:
        """
        Return all reachable states for pair (state, a) given.
        :param state:
        :param action:
        :return:
        """

        # Set current state with state indicated
        self.current_state = state

        # Get all actions available
        actions = self.action_space.copy()

        # Return all possible states reachable with any action
        return {self.next_state(action=a, state=state) for a in actions}
