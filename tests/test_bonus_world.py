"""
Unit tests file where testing BonusWorld environment.
"""

import unittest

from gym import spaces

from gym_tiadas.gym_tiadas.envs import BonusWorld


class TestBonusWorld(unittest.TestCase):
    environment = None

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = BonusWorld(seed=0)

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
        self.assertTrue(hasattr(self.environment, 'pits'))
        self.assertTrue(hasattr(self.environment, 'bonus'))
        self.assertTrue(hasattr(self.environment, 'bonus_activated'))

        # By default mesh shape is 9x9
        self.assertEqual(spaces.Tuple((spaces.Discrete(9), spaces.Discrete(9))), self.environment.observation_space)

        # By default action space is 4 (UP, RIGHT, DOWN, LEFT)
        self.assertEqual(spaces.Discrete(4), self.environment.action_space)

        # By default initial state is (0, 0)
        self.assertEqual((0, 0), self.environment.initial_state)
        self.assertEqual(self.environment.initial_state, self.environment.current_state)

        # Default reward is (0, 0, -1)
        self.assertEqual((0, 0, -1), self.environment.default_reward)

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
        # Set bonus activated to True
        self.environment.bonus_activated = True

        # Reset environment
        self.environment.reset()

        # Asserts
        self.assertEqual(self.environment.initial_state, self.environment.current_state)
        self.assertFalse(self.environment.bonus_activated)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        ################################################################################################################
        # Begin at state (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        state = (0, 0)

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
        # Set to (8, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (8, 0)
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
        # Set to (8, 8) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (8, 8)
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

        # Go to LEFT (decrement x axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual((state[0] - 1, state[1]), new_state)

        ################################################################################################################
        # Set to (0, 8) (DOWN-LEFT corner)
        ################################################################################################################
        state = (0, 8)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('UP'))
        self.assertEqual((state[0], state[1] - 1), new_state)

        # Go to RIGHT (increment x axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual((state[0] + 1, state[1]), new_state)

        # Cannot go to DOWN (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual(state, new_state)

        # Cannot go to LEFT (Keep in same state
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual(state, new_state)

        ################################################################################################################
        # Obstacles (For example, (2, 2)
        ################################################################################################################

        # Set to (2, 1)
        state = (2, 1)
        self.environment.current_state = state

        # Cannot go to DOWN (Keep in same state), because there is an obstacle.
        new_state = self.environment.next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual(state, new_state)

        # Set to (1, 2)
        state = (1, 2)
        self.environment.current_state = state

        # Cannot go to RIGHT (Keep in same state), because there is an obstacle.
        new_state = self.environment.next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual(state, new_state)

        # Set to (2, 1)
        state = (2, 1)
        self.environment.current_state = state

        # Cannot go to DOWN (Keep in same state), because there is an obstacle.
        new_state = self.environment.next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual(state, new_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Simple valid step, at each step penalizes -1.
        new_state, rewards, is_final, info = self.environment.step(action=self.environment.actions.get('DOWN'))

        # Remember that initial state is (0, 0)
        self.assertEqual((0, 1), new_state)
        self.assertEqual([0, 0, -1], rewards)
        self.assertFalse(is_final)
        self.assertFalse(info)

        # Do 7 steps more to reach final step (0, 8), which reward is (9, 1)
        for _ in range(7):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        self.assertEqual((0, 8), new_state)
        self.assertEqual([9, 1, -1], rewards)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final step
        for _ in range(8):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        self.assertEqual((8, 0), new_state)
        self.assertEqual([1, 9, -1], rewards)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to a PIT state (and reset to initial_state)
        _, _, _, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        for _ in range(7):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        self.assertEqual((0, 0), new_state)
        self.assertEqual([0, 0, -1], rewards)
        self.assertFalse(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # At first bonus is disabled
        self.assertFalse(self.environment.bonus_activated)

        # Get bonus and go to final state

        # 4 steps to RIGHT
        for _ in range(4):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        # 3 steps to DOWN
        for _ in range(3):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        # 1 step to LEFT
        for _ in range(1):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('LEFT'))

        # Now bonus is activated
        self.assertTrue(self.environment.bonus_activated)

        # Go to final state with bonus activated

        # 3 steps to DOWN
        for _ in range(3):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        # 1 step to RIGHT
        for _ in range(1):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        # 2 steps to DOWN
        for _ in range(2):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        # Final state (4, 8), which reward is (9, 5), but bonus is activated.
        self.assertEqual((4, 8), new_state)
        self.assertEqual([9 * 2, 5 * 2, -1], rewards)
        self.assertTrue(is_final)
