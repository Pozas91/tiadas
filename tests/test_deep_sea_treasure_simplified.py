"""
Unit tests file where testing DeepSeaTreasureSimplified environment.
"""

import unittest

from gym import spaces

from gym_tiadas.gym_tiadas.envs import DeepSeaTreasureSimplified


class TestDeepSeaTreasureSimplified(unittest.TestCase):
    environment = None

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = DeepSeaTreasureSimplified(seed=0)

    def tearDown(self):
        self.environment = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # By default mesh shape is 10x11
        self.assertEqual(spaces.Tuple((spaces.Discrete(3), spaces.Discrete(4))), self.environment.observation_space)

        # By default action space is 4 (UP, RIGHT, DOWN, LEFT)
        self.assertIsInstance(self.environment.action_space, spaces.Space)

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
        # Set to (2, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (2, 0)
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
        # Set to (2, 3) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (2, 3)
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

        # Simple valid step, begin at state (0, 0)
        new_state, rewards, is_final, info = self.environment.step(action=self.environment.actions.get('DOWN'))

        # Reward:
        #   [time_inverted, treasure_value]

        self.assertEqual((0, 1), new_state)
        self.assertEqual([-1, 5], rewards)
        self.assertTrue(is_final)
        self.assertFalse(info)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final state.
        # Go to right 2 steps, until (2, 0).
        for _ in range(2):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        # Go to down 3 steps, until (2, 3).
        for _ in range(3):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        self.assertEqual((2, 3), new_state)
        self.assertEqual([-1, 120], rewards)
        self.assertTrue(is_final)
