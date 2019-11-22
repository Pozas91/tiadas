"""
Unit tests file where testing DeepSeaTreasure environment.
"""

import utils.environments as ue
from environments import PyramidMDP
from tests.environments.test_env_mesh import TestEnvMesh


class TestPyramidMDP(TestEnvMesh):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = PyramidMDP(seed=0)

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        super().__init__()

        # This environment must have another attributes
        self.assertTrue(hasattr(self.environment, 'n_transition'))

    def test_action_space_length(self):
        # By default action space is 4 (UP, RIGHT, DOWN, LEFT)
        self.assertEqual(self.environment.action_space.n, 4)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        # Disable stochastic transition
        self.environment.n_transition = 1.

        ################################################################################################################
        # Begin at position (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        state = (0, 0)
        self.environment.current_state = state

        # Cannot go to UP (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(state, new_state)

        # Go to RIGHT (increment x axis)
        new_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((state[0] + 1, state[1]), new_state)

        # Go to DOWN (increment y axis)
        new_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((state[0], state[1] + 1), new_state)

        # Cannot go to LEFT (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(state, new_state)

        ################################################################################################################
        # Set to (9, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (9, 0)
        self.environment.current_state = state

        # Cannot go to UP (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(state, new_state)

        # Cannot go to RIGHT (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(state, new_state)

        # Cannot go to DOWN (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(state, new_state)

        # Go to LEFT (decrement x-axis)
        new_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual((state[0] - 1, state[1]), new_state)

        ################################################################################################################
        # Set to (0, 9) (DOWN-LEFT corner)
        ################################################################################################################
        state = (0, 9)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        new_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual((state[0], state[1] - 1), new_state)

        # Cannot go to RIGHT (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(state, new_state)

        # Cannot go to DOWN (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(state, new_state)

        # Cannot go to LEFT (Keep in same position, obstacle)
        new_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(state, new_state)

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

        self.assertEqual((1, 0), next_state)
        self.assertEqual([-1, -1], reward)
        self.assertFalse(is_final)

        for _ in range(5):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # Remember that initial position is (0, 0)
        self.assertEqual((1, 0), next_state)
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

                # At least 3 states are available
                self.assertTrue(3 <= len(reachable_states) <= 4)

                self.assertTrue(
                    # Or go to UP
                    (x, y - 1) in reachable_states or
                    # Or go to left
                    (x - 1, y) in reachable_states or
                    # Or got to right
                    (x + 1, y) in reachable_states or
                    # Or got to down
                    (x, y + 1) in reachable_states or
                    # Or keep in same position
                    (x, y) in reachable_states
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
