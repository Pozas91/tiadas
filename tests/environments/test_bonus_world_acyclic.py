"""
Unit tests file where testing BonusWorld environment.
"""

from environments import BonusWorldAcyclic
from tests.environments.test_bonus_world import TestBonusWorld


class TestBonusWorldAcyclic(TestBonusWorld):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = BonusWorldAcyclic(seed=0)

    def test_action_space_length(self):
        # By default action space is 2 (RIGHT, DOWN)
        self.assertEqual(len(self.environment.actions), 2)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        ################################################################################################################
        # Begin at position (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        position = (0, 0)
        self.environment.current_state = position, False

        action_space = self.environment.action_space

        self.assertTrue(
            action_space.n == 2 and action_space.contains(
                self.environment.actions['DOWN']
            ) and action_space.contains(
                self.environment.actions['RIGHT']
            )
        )

        # Go to RIGHT (increment x axis)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((position[0] + 1, position[1]), next_state[0])

        # Go to DOWN (increment y axis)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((position[0], position[1] + 1), next_state[0])

        ################################################################################################################
        # Set to (8, 0) (TOP-RIGHT corner)
        ################################################################################################################
        position = (8, 0)
        self.environment.current_state = position, False

        action_space = self.environment.action_space

        self.assertTrue(action_space.n == 1 and action_space.contains(self.environment.actions['DOWN']))

        # Go to DOWN (increment y axis)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((position[0], position[1] + 1), next_state[0])

        ################################################################################################################
        # Set to (8, 8) (DOWN-RIGHT corner)
        ################################################################################################################
        position = (8, 8)
        self.environment.current_state = position, False

        action_space = self.environment.action_space

        self.assertTrue(action_space.n == 0)

        ################################################################################################################
        # Set to (0, 8) (DOWN-LEFT corner)
        ################################################################################################################
        position = (0, 8)
        self.environment.current_state = position, False

        action_space = self.environment.action_space

        self.assertTrue(action_space.n == 1 and action_space.contains(self.environment.actions['RIGHT']))

        ################################################################################################################
        # Obstacles (For example, (2, 2)
        ################################################################################################################

        # Set to (2, 1)
        position = (2, 1)
        self.environment.current_state = position, False

        action_space = self.environment.action_space

        self.assertTrue(action_space.n == 1 and action_space.contains(self.environment.actions['RIGHT']))

        # Set to (1, 2)
        position = (1, 2)
        self.environment.current_state = position, False

        action_space = self.environment.action_space

        self.assertTrue(action_space.n == 1 and action_space.contains(self.environment.actions['DOWN']))

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Simple valid step, at each step penalizes -1.
        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['DOWN'])

        # Remember that initial position is (0, 0)
        self.assertEqual(((0, 1), False), next_state)
        self.assertEqual([0, 0, -1], reward)
        self.assertFalse(is_final)
        self.assertFalse(info)

        # Do 7 steps more to reach final step (0, 8), which reward is (9, 1)
        for _ in range(7):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual(((0, 8), False), next_state)
        self.assertEqual([9, 1, -1], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final step
        for _ in range(8):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual(((8, 0), False), next_state)
        self.assertEqual([1, 9, -1], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to a PIT position (and reset to initial_state)
        _, _, _, _ = self.environment.step(action=self.environment.actions['DOWN'])

        for _ in range(7):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual(((7, 1), False), next_state)
        self.assertEqual([-50, -50, -1], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # At first bonus is disabled
        self.assertFalse(self.environment.current_state[1])

        # Get bonus and go to final position

        # 3 steps to RIGHT
        for _ in range(3):
            _, _, _, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # 3 steps to DOWN
        for _ in range(3):
            _, _, _, _ = self.environment.step(action=self.environment.actions['DOWN'])

        # Now bonus is activated
        self.assertTrue(self.environment.observation_space[1])

        # Go to final position with bonus activated

        # 3 steps to DOWN
        for _ in range(3):
            _, _, _, _ = self.environment.step(action=self.environment.actions['DOWN'])

        # 1 step to RIGHT
        for _ in range(1):
            _, _, _, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # 2 steps to DOWN
        for _ in range(2):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        # Final position (4, 8), which reward is (9, 5), but bonus is activated.
        self.assertEqual(((4, 8), True), next_state)
        self.assertEqual([9 * 2, 5 * 2, -1], reward)
        self.assertTrue(is_final)

    def test_states_size(self):
        # Same that original problem, but without pits (are final states)
        self.assertEqual(len(self.environment.states()), 129)

    def test_reachable_states(self):

        with self.assertRaises(ValueError):
            # Obstacle on right
            self.environment.reachable_states(((1, 2), False), action=self.environment.actions['RIGHT'])

            # Obstacle down
            self.environment.reachable_states(((2, 1), False), action=self.environment.actions['DOWN'])

        reachable_states = self.environment.reachable_states(
            self.environment.initial_state, action=self.environment.actions['RIGHT']
        )

        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(((1, 0), False), reachable_states)

        reachable_states = self.environment.reachable_states(
            self.environment.initial_state, action=self.environment.actions['DOWN']
        )

        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(((0, 1), False), reachable_states)

        # Go to bonus (in this variant isn't obstacle to go right)
        reachable_states = self.environment.reachable_states(
            ((2, 3), False), action=self.environment.actions['RIGHT']
        )

        # Bonus has been activated
        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(((3, 3), True), reachable_states)

        # Go to PIT
        reachable_states = self.environment.reachable_states(
            ((0, 7), False), action=self.environment.actions['RIGHT']
        )

        # Doesn't reset to initial state (it's a final state)
        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(((1, 7), False), reachable_states)

        # Go to PIT, with bonus enabled
        reachable_states = self.environment.reachable_states(
            ((0, 7), True), action=self.environment.actions['RIGHT']
        )

        # Reset to initial state
        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(((1, 7), True), reachable_states)

        # Go to a final state
        reachable_states = self.environment.reachable_states(
            ((0, 7), True), action=self.environment.actions['DOWN']
        )

        # Reset to initial state
        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(((0, 8), True), reachable_states)
