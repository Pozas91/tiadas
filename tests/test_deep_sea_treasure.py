"""
Unit tests file where testing DeepSeaTreasure environment.
"""

import unittest

from gym import spaces

from gym_tiadas.gym_tiadas.envs import DeepSeaTreasure


class TestDeepSeaTreasure(unittest.TestCase):
    environment = None

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = DeepSeaTreasure(seed=0)

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

        # This environment must have another attributes
        self.assertTrue(hasattr(self.environment, 'finals'))
        self.assertTrue(hasattr(self.environment, 'obstacles'))

        # By default mesh shape is 10x11
        self.assertEqual(spaces.Tuple((spaces.Discrete(10), spaces.Discrete(11))), self.environment.observation_space)

        # By default action space is 4 (UP, RIGHT, DOWN, LEFT)
        self.assertEqual(spaces.Discrete(4), self.environment.action_space)

        # By default initial state is (0, 0)
        self.assertEqual((0, 0), self.environment.initial_state)
        self.assertEqual(self.environment.initial_state, self.environment.current_state)

        # Default reward is (-1, 0)
        self.assertEqual((-1, 0), self.environment.default_reward)

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

        ################################################################################################################
        # Begin at state (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        state = (0, 0)
        self.environment.current_state = state

        # Cannot go to UP (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('UP'))
        self.assertEqual(state, new_state)

        # Go to RIGHT (increment x axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual((state[0] + 1, state[1]), new_state)

        # Go to DOWN (increment y axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual((state[0], state[1] + 1), new_state)

        # Cannot go to LEFT (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual(state, new_state)

        ################################################################################################################
        # Set to (9, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (9, 0)
        self.environment.current_state = state

        # Cannot go to UP (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('UP'))
        self.assertEqual(state, new_state)

        # Cannot go to RIGHT (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual(state, new_state)

        # Go to DOWN (increment y axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual((state[0], state[1] + 1), new_state)

        # Go to LEFT (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual((state[0] - 1, state[1]), new_state)

        ################################################################################################################
        # Set to (9, 10) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (9, 10)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('UP'))
        self.assertEqual((state[0], state[1] - 1), new_state)

        # Cannot go to RIGHT (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual(state, new_state)

        # Cannot go to DOWN (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual(state, new_state)

        # Cannot go to LEFT (Keep in same state, obstacle)
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual(state, new_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Simple valid step
        # Reward:
        #   [time_inverted, treasure_value]
        new_state, rewards, is_final, info = self.environment.step(action=self.environment.actions.get('DOWN'))

        # Remember that initial state is (0, 0)
        self.assertEqual((0, 1), new_state)
        self.assertEqual([-1, 1], rewards)
        self.assertTrue(is_final)
        self.assertFalse(info)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final state.
        # Go to right 6 steps, until (6, 0).
        for _ in range(6):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        # Go to down 6 steps, until (6, 6).
        for _ in range(6):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        # Try to go LEFT, but is an obstacle.
        new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('LEFT'))

        self.assertEqual((6, 6), new_state)
        self.assertEqual([-1, 0], rewards)
        self.assertFalse(is_final)

        # Go to DOWN to get 24 treasure-value.
        new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        self.assertEqual((6, 7), new_state)
        self.assertEqual([-1, 24], rewards)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final state.
        # Go to right 9 steps, until (9, 0).
        for _ in range(9):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        # Go to down 10 steps, until (9, 10).
        for _ in range(10):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        self.assertEqual((9, 10), new_state)
        self.assertEqual([-1, 124], rewards)
        self.assertTrue(is_final)
