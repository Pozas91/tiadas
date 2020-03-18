"""
Unit tests path where testing DeepSeaTreasure environment.
"""

import utils.environments as ue
from environments import PyramidMDPNoBounces
from tests.environments.test_pyramid_mdp import TestPyramidMDP


class TestPyramidMDPNoBounces(TestPyramidMDP):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = PyramidMDPNoBounces(seed=0)

    def test_action_space_length(self):
        # By default action space is 2 (RIGHT, DOWN)
        self.assertEqual(self.environment.action_space.n, 2)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Disable stochastic transitions
        self.environment.n_transition = 1.

        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['DOWN'])

        # Remember that initial position is (0, 0)
        self.assertEqual((0, 1), next_state)
        self.assertEqual([-1, -1], reward)
        self.assertFalse(is_final)
        self.assertFalse(info)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final position.
        # Go to right 9 steps, until (9, 0).
        for _ in range(9):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual((9, 0), next_state)
        self.assertEqual([100, 10], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final position.
        # Go to down 9 steps, until (0, 9).
        for _ in range(9):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual((0, 9), next_state)
        self.assertEqual([10, 100], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to an intermediate state
        for _ in range(4):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual((0, 4), next_state)
        self.assertEqual([-1, -1], reward)
        self.assertFalse(is_final)

        for _ in range(5):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # Remember that initial position is (0, 0)
        self.assertEqual((5, 4), next_state)
        self.assertEqual([60, 50], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Enable stochastic transitions
        self.environment.n_transition = 0.

        # Go to an intermediate state
        for _ in range(4):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual((0, 0), next_state)
        self.assertEqual([-1, -1], reward)
        self.assertFalse(is_final)

        for _ in range(5):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # Remember that initial position is (0, 0)
        self.assertEqual((2, 1), next_state)
        self.assertEqual([-1, -1], reward)
        self.assertFalse(is_final)

    def test_states_size(self):
        self.assertEqual(len(self.environment.states()), 45)

    def test_reachable_states(self):

        # For any state the following happens
        for state in self.environment.states():

            # Decompose state
            x, y = state

            # Set current state
            self.environment.current_state = state

            for action in self.environment.action_space:
                # If go to UP, or can go to UP or keep in same position
                reachable_states = self.environment.reachable_states(state=state, action=action)
                reachable_states_len = len(reachable_states)

                expected_reachable_states = {
                    (x, y - 1), (x - 1, y), (x + 1, y), (x, y + 1)
                }
                expected_reachable_states_len = 4

                if x <= 0:
                    expected_reachable_states.remove((x - 1, y))
                    expected_reachable_states_len -= 1

                if y <= 0:
                    expected_reachable_states.remove((x, y - 1))
                    expected_reachable_states_len -= 1

                # Check if has correct length
                self.assertEqual(expected_reachable_states_len, reachable_states_len)

                self.assertTrue(
                    all(
                        element in expected_reachable_states for element in reachable_states
                    ) and
                    all(
                        element in reachable_states for element in expected_reachable_states
                    )
                )

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

                    if (
                            ue.is_on_up_or_same_position(state=state, next_state=next_state) and
                            (action == self.environment.actions['UP'])
                    ) or (
                            ue.is_on_right_or_same_position(state=state, next_state=next_state) and
                            (action == self.environment.actions['RIGHT'])
                    ) or (
                            ue.is_on_down_or_same_position(state=state, next_state=next_state) and
                            (action == self.environment.actions['DOWN'])
                    ) or (
                            ue.is_on_left_or_same_position(state=state, next_state=next_state) and
                            (action == self.environment.actions['LEFT'])
                    ):
                        self.assertEqual(self.environment.n_transition, probability)
                    else:
                        self.assertEqual(
                            (1. - self.environment.n_transition) / self.environment.action_space.n, probability
                        )

    def test_transition_reward(self):

        for state in self.environment.states():

            self.environment.current_state = state

            for action in self.environment.action_space:

                for next_state in self.environment.reachable_states(state=state, action=action):

                    reward = self.environment.transition_reward(state=state, action=action, next_state=next_state)

                    if self.environment.is_final(state=next_state):
                        self.assertEqual(((next_state[0] + 1) * 10, (next_state[1] + 1) * 10), reward)
                    else:
                        self.assertEqual([-1, -1], reward)
