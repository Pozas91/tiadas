"""
Unit tests file where testing NonRecurrentRings environment.
"""

import unittest

from gym import spaces

from gym_tiadas.gym_tiadas.envs import NonRecurrentRings


class TestNonRecurrentRings(unittest.TestCase):
    environment = None

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = NonRecurrentRings(seed=0)

    def tearDown(self):
        self.environment = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # Observation space is 8 states
        self.assertEqual(spaces.Discrete(8), self.environment.observation_space)

        # By default action space is 2 (CLOCKWISE, COUNTER-CLOCKWISE)
        self.assertIsInstance(self.environment.action_space, spaces.Space)

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
        new_state = self.environment.next_state(action=self.environment.actions.get('CLOCKWISE'))

        # State 8
        self.assertEqual(7, new_state)

        ################################################################################################################

        # Set state
        self.environment.current_state = 7

        # Go to counter-clockwise sense
        new_state = self.environment.next_state(action=self.environment.actions.get('COUNTER-CLOCKWISE'))

        # State 1
        self.assertEqual(0, new_state)

        ################################################################################################################

        # Set state
        self.environment.current_state = 0

        # Go to counter-clockwise sense
        new_state = self.environment.next_state(action=self.environment.actions.get('COUNTER-CLOCKWISE'))

        # State 2
        self.assertEqual(1, new_state)

        ################################################################################################################

        # Set state
        self.environment.current_state = 1

        # Go to counter-clockwise sense
        new_state = self.environment.next_state(action=self.environment.actions.get('COUNTER-CLOCKWISE'))

        # State 3
        self.assertEqual(2, new_state)

        ################################################################################################################

        # Set state
        self.environment.current_state = 3

        # Go to counter-clockwise sense from state 3.
        new_state = self.environment.next_state(action=self.environment.actions.get('COUNTER-CLOCKWISE'))

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
        self.assertEqual((2, -1), rewards)
        self.assertFalse(is_final)
        self.assertFalse(info)

        ################################################################################################################

        # State 2
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('COUNTER-CLOCKWISE')
        )

        self.assertEqual(2, new_state)
        self.assertEqual((2, -1), rewards)

        # State 3
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('COUNTER-CLOCKWISE')
        )

        self.assertEqual(3, new_state)
        self.assertEqual((2, -1), rewards)

        # State 4
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('COUNTER-CLOCKWISE')
        )

        self.assertEqual(0, new_state)
        self.assertEqual((2, -1), rewards)

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

        self.assertEqual(7, new_state)
        self.assertEqual((-1, 0), rewards)

        # State 8
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('CLOCKWISE')
        )

        self.assertEqual(4, new_state)
        self.assertEqual((-1, 2), rewards)

        # State 5
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('CLOCKWISE')
        )

        self.assertEqual(5, new_state)
        self.assertEqual((-1, 2), rewards)

        # State 6
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('CLOCKWISE')
        )

        self.assertEqual(6, new_state)
        self.assertEqual((-1, 2), rewards)

        # State 7
        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('CLOCKWISE')
        )

        self.assertEqual(7, new_state)
        self.assertEqual((-1, 2), rewards)

        # State 8
        _ = self.environment.step(
            action=self.environment.actions.get('CLOCKWISE')
        )

        new_state, rewards, is_final, info = self.environment.step(
            action=self.environment.actions.get('COUNTER-CLOCKWISE')
        )

        self.assertEqual(7, new_state)
        self.assertEqual((0, -1), rewards)
