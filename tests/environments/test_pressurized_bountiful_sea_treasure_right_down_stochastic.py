"""
Unit tests path where testing test PressurizedBountifulSeaTreasure environment.
"""

import utils.environments as ue
from environments import PressurizedBountifulSeaTreasureRightDownStochastic
from tests.environments.test_pressurized_bountiful_sea_treasure import TestPressurizedBountifulSeaTreasure


class TestPressurizedBountifulSeaTreasureRightDownStochastic(TestPressurizedBountifulSeaTreasure):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = PressurizedBountifulSeaTreasureRightDownStochastic(seed=0)

    def test_action_space_length(self):
        # By default action space is 2 (RIGHT, DOWN)
        self.assertEqual(len(self.environment.actions), 3)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        # Null stochastic transitions
        self.environment.p_stochastic = 1.

        ################################################################################################################
        # Begin at position (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        state = (0, 0)
        self.environment.current_state = state

        action_space = self.environment.action_space

        self.assertTrue(
            action_space.n == 2 and action_space.contains(
                self.environment.actions['DOWN_PROB']
            ) and action_space.contains(
                self.environment.actions['RIGHT_PROB']
            )
        )

        # Go to RIGHT (increment x axis)
        new_state = self.environment.next_state(action=self.environment.actions['RIGHT_PROB'])
        self.assertEqual((state[0] + 1, state[1]), new_state)

        # Go to DOWN (increment y axis)
        new_state = self.environment.next_state(action=self.environment.actions['DOWN_PROB'])
        self.assertEqual((state[0], state[1] + 1), new_state)

        ################################################################################################################
        # Set to (9, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (9, 0)
        self.environment.current_state = state

        # Cannot go to RIGHT (Keep in same position)
        with self.assertRaises(ValueError):
            self.environment.next_state(action=self.environment.actions['RIGHT_PROB'])
            self.environment.next_state(action=self.environment.actions['DOWN_PROB'])

        # Go to DOWN (increment y axis)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((state[0], state[1] + 1), next_state)

        ################################################################################################################
        # Set to (9, 10) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (9, 10)
        self.environment.current_state = state

        with self.assertRaises(ValueError):
            self.environment.next_state(action=self.environment.actions['RIGHT_PROB'])
            self.environment.next_state(action=self.environment.actions['DOWN_PROB'])
            self.environment.next_state(action=self.environment.actions['DOWN'])

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Disable p_stochastic
        self.environment.p_stochastic = 1.

        # Simple valid step
        # Reward:
        #   [time_inverted, treasure_value]
        # Remember that initial position is (0, 0)

        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual((0, 1), next_state)
        self.assertEqual([-1, 5, -2], reward)
        self.assertTrue(is_final)
        self.assertFalse(info)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Simple valid step
        # Reward:
        #   [time_inverted, treasure_value]
        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['DOWN_PROB'])

        # Remember that initial position is (0, 0)
        self.assertEqual((0, 1), next_state)
        self.assertEqual([-1, 5, -2], reward)
        self.assertTrue(is_final)
        self.assertFalse(info)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final position.
        # Go to right 6 steps, until (6, 0).
        for _ in range(6):
            _, _, _, _ = self.environment.step(action=self.environment.actions['RIGHT_PROB'])

        # Go to down 7 steps, until (6, 7).
        for _ in range(7):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual((6, 7), next_state)
        self.assertEqual([-1, 163, -8], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final position.
        # Go to right 6 steps, until (6, 0).
        for _ in range(6):
            _, _, _, _ = self.environment.step(action=self.environment.actions['RIGHT_PROB'])

        # Go to down 7 steps, until (6, 7).
        for _ in range(7):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN_PROB'])

        self.assertEqual((6, 7), next_state)
        self.assertEqual([-1, 163, -8], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final position.
        # Go to right 9 steps, until (9, 0).
        for _ in range(9):
            _, _, _, _ = self.environment.step(action=self.environment.actions['RIGHT_PROB'])

        # Go to down 10 steps, until (9, 10).
        for _ in range(10):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual((9, 10), next_state)
        self.assertEqual([-1, 175, -11], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final position.
        # Go to right 9 steps, until (9, 0).
        for _ in range(9):
            _, _, _, _ = self.environment.step(action=self.environment.actions['RIGHT_PROB'])

        # Go to down 10 steps, until (9, 10).
        for _ in range(10):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN_PROB'])

        self.assertEqual((9, 10), next_state)
        self.assertEqual([-1, 175, -11], reward)
        self.assertTrue(is_final)

    def test_transition_probability(self):

        # For all states, for all actions and for all next_state possibles, transition probability must be return 1.
        for state in self.environment.states():

            # Set state as current state
            self.environment.current_state = state

            for action in self.environment.action_space:

                for next_state in self.environment.reachable_states(state=state, action=action):

                    probability = self.environment.transition_probability(
                        state=state, action=action, next_state=next_state
                    )

                    if action == self.environment.actions['RIGHT_PROB']:

                        if ue.is_on_right_or_same_position(state=state, next_position=next_state):
                            self.assertEqual(1. - self.environment.p_stochastic, probability)
                        else:
                            self.assertEqual(self.environment.p_stochastic, probability)

                    elif action == self.environment.actions['DOWN_PROB']:

                        if ue.is_on_down_or_same_position(state=state, next_state=next_state):
                            self.assertEqual(1. - self.environment.p_stochastic, probability)
                        else:
                            self.assertEqual(self.environment.p_stochastic, probability)

                    elif action == self.environment.actions['DOWN']:
                        self.assertEqual(1., probability)
                    else:
                        raise ValueError('Action invalid.')

    def test_reachable_states(self):

        # For any state the following happens
        for state in self.environment.states():
            # Decompose state
            x, y = state

            if x < (self.environment.observation_space[0].n - 1):
                # Go to right
                reachable_states = self.environment.reachable_states(
                    state=state, action=self.environment.actions['RIGHT_PROB']
                )

                self.assertEqual(2, len(reachable_states))
                self.assertEqual(
                    reachable_states - {(x + 1, y), (x, y + 1)}, {(x + 1, y), (x, y + 1)} - reachable_states
                )

                # Go to down_prob
                reachable_states = self.environment.reachable_states(
                    state=state, action=self.environment.actions['DOWN_PROB']
                )

                self.assertTrue(2, len(reachable_states))
                self.assertEqual(
                    reachable_states - {(x + 1, y), (x, y + 1)}, {(x + 1, y), (x, y + 1)} - reachable_states
                )

            else:
                with self.assertRaises(ValueError):
                    self.environment.reachable_states(state=state, action=self.environment.actions['RIGHT_PROB'])
                    self.environment.reachable_states(state=state, action=self.environment.actions['DOWN_PROB'])

            # Go to down
            reachable_states = self.environment.reachable_states(state=state, action=self.environment.actions['DOWN'])

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x, y + 1)})
