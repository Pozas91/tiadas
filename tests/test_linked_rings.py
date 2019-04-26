"""
Unit tests file where testing LinkedRings environment.
"""

import unittest

from gym import spaces

from gym_tiadas.gym_tiadas.envs import LinkedRings


class TestLinkedRings(unittest.TestCase):
    environment = None

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = LinkedRings(seed=0)

    def tearDown(self):
        self.environment = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # All environments must be have an action_space and an observation_space attributes.
        self.assertTrue(hasattr(self.environment, 'action_space'))
        self.assertTrue(hasattr(self.environment, 'observation_space'))

        # All environments must be have an step, seed, reset, render and _next_state methods.
        self.assertTrue(hasattr(self.environment, 'step'))
        self.assertTrue(hasattr(self.environment, 'seed'))
        self.assertTrue(hasattr(self.environment, 'reset'))
        self.assertTrue(hasattr(self.environment, '_next_state'))

        # Observation space is 7 states
        self.assertEqual(spaces.Discrete(7), self.environment.observation_space)

        # By default action space is 2 (CLOCKWISE, COUNTER-CLOCKWISE)
        self.assertEqual(spaces.Discrete(2), self.environment.action_space)

        # By default initial state is 0
        self.assertEqual(0, self.environment.initial_state)
        self.assertEqual(self.environment.initial_state, self.environment.current_state)

    def test_seed(self):
        """
        Testing seed method
        :return:
        """
        self.environment.seed(seed=0)
        n1_1 = self.environment.np_random.randint(0, 10)
        n1_2 = self.environment.np_random.randint(0, 10)

        self.environment.seed(seed=0)
        n2_1 = self.environment.np_random.randint(0, 10)
        n2_2 = self.environment.np_random.randint(0, 10)

        self.assertEqual(n1_1, n2_1)
        self.assertEqual(n1_2, n2_2)

    def test_reset(self):
        """
        Testing reset method
        :return:
        """

        # Set current state to random state
        self.environment.current_state = self.environment.observation_space.sample()

        # Reset environment
        self.environment.reset()

        # Asserts
        self.assertEqual(self.environment.initial_state, self.environment.current_state)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        # Set state
        self.environment.current_state = 0

        # Go to counter-clockwise sense
        new_state = self.environment._next_state(action=self.environment.actions.get('CLOCKWISE'))

        # State 5
        self.assertEqual(4, new_state)

        ################################################################################################################

        # Set state
        self.environment.current_state = 4

        # Go to counter-clockwise sense
        new_state = self.environment._next_state(action=self.environment.actions.get('COUNTER-CLOCKWISE'))

        # State 1
        self.assertEqual(0, new_state)

        ################################################################################################################

        # Set state
        self.environment.current_state = 0

        # Go to counter-clockwise sense
        new_state = self.environment._next_state(action=self.environment.actions.get('COUNTER-CLOCKWISE'))

        # State 2
        self.assertEqual(1, new_state)

        ################################################################################################################

        # Set state
        self.environment.current_state = 1

        # Go to counter-clockwise sense
        new_state = self.environment._next_state(action=self.environment.actions.get('COUNTER-CLOCKWISE'))

        # State 3
        self.assertEqual(2, new_state)

        ################################################################################################################

        # Set state
        self.environment.current_state = 3

        # Go to counter-clockwise sense from state 3.
        new_state = self.environment._next_state(action=self.environment.actions.get('COUNTER-CLOCKWISE'))

        # State 1
        self.assertEqual(0, new_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # THIS ENVIRONMENT DOES NOT HAVE FINAL STEP (is_final is always False)
        # Reward:
        #   [value_1, value_2]

        # Simple valid step, begin at state 0
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('COUNTER-CLOCKWISE')
        )

        self.assertEqual(1, new_state)
        self.assertEqual((3, -1), rewards)
        self.assertFalse(is_final)
        self.assertFalse(info)

        ################################################################################################################

        # State 2
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('COUNTER-CLOCKWISE')
        )

        self.assertEqual(2, new_state)
        self.assertEqual((3, -1), rewards)

        # State 3
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('COUNTER-CLOCKWISE')
        )

        self.assertEqual(3, new_state)
        self.assertEqual((3, -1), rewards)

        # State 4
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('COUNTER-CLOCKWISE')
        )

        self.assertEqual(0, new_state)
        self.assertEqual((3, -1), rewards)

        # State 1
        _ = self.environment.step(
            action=self.environment.actions.get('COUNTER-CLOCKWISE')
        )

        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('CLOCKWISE')
        )

        self.assertEqual(0, new_state)
        self.assertEqual((-1, 0), rewards)

        # State 1
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('CLOCKWISE')
        )

        self.assertEqual(4, new_state)
        self.assertEqual((-1, 3), rewards)

        # State 5
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('CLOCKWISE')
        )

        self.assertEqual(5, new_state)
        self.assertEqual((-1, 3), rewards)

        # State 6
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('CLOCKWISE')
        )

        self.assertEqual(6, new_state)
        self.assertEqual((-1, 3), rewards)

        # State 7
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('CLOCKWISE')
        )

        self.assertEqual(0, new_state)
        self.assertEqual((-1, 3), rewards)

        # State 1
        _ = self.environment.step(
            action=self.environment.actions.get('CLOCKWISE')
        )

        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('COUNTER-CLOCKWISE')
        )

        self.assertEqual(0, new_state)
        self.assertEqual((0, -1), rewards)
