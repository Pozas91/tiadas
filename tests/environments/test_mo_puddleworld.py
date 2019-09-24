"""
Unit tests file where testing MoPuddleWorld environment.
"""

import unittest

from gym import spaces

from environments import MoPuddleWorld


class TestMoPuddleWorld(unittest.TestCase):
    environment = None

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = MoPuddleWorld(seed=0)

    def tearDown(self):
        self.environment = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # By default mesh shape is 20x20
        self.assertEqual(spaces.Tuple((spaces.Discrete(20), spaces.Discrete(20))), self.environment.observation_space)

        # By default action space is 4 (UP, RIGHT, DOWN, LEFT)
        self.assertIsInstance(self.environment.action_space, spaces.Space)

        # Default reward is (10, 0)
        self.assertEqual((10, 0), self.environment.default_reward)

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

        # Reset environment
        self.environment.reset()

        # Asserts
        self.assertTrue(self.environment.observation_space.contains(self.environment.current_state))
        self.assertFalse(self.environment.current_state == self.environment.final_state)

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
        # Set to (19, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (19, 0)
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
        # Set to (19, 19) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (19, 19)
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

        # Cannot go to LEFT (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual((state[0] - 1, state[1]), new_state)

        ################################################################################################################
        # Set to (0, 19) (DOWN-LEFT corner)
        ################################################################################################################
        state = (0, 19)
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

        # Cannot go to LEFT (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual(state, new_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # THIS ENVIRONMENT HAS A RANDOM INITIAL STATE. FOR TESTING I USE A PREDEFINED INITIAL STATE.
        # Reward:
        #   [non_goal_reached, puddle_penalize]

        new_state = None
        rewards = None
        is_final = False
        info = {}

        # Set a current state
        self.environment.current_state = (0, 0)

        # Simple valid step
        for _ in range(3):
            new_state, rewards, is_final, info = self.environment.step(action=self.environment.actions.get('DOWN'))

        # Remember that initial state is (0, 0)
        self.assertEqual((0, 3), new_state)
        self.assertEqual([-1, -1], rewards)
        self.assertFalse(is_final)
        self.assertFalse(info)

        # Enter in puddles a little more.
        new_state, rewards, is_final, info = self.environment.step(action=self.environment.actions.get('DOWN'))

        self.assertEqual((0, 4), new_state)
        self.assertEqual([-1, -2], rewards)

        # Enter in puddles a little more.
        new_state, rewards, is_final, info = self.environment.step(action=self.environment.actions.get('DOWN'))

        self.assertEqual((0, 5), new_state)
        self.assertEqual([-1, -2], rewards)

        # Enter a more...
        for _ in range(7):
            new_state, rewards, is_final, info = self.environment.step(action=self.environment.actions.get('RIGHT'))

        self.assertEqual((7, 5), new_state)
        self.assertEqual([-1, -4], rewards)

        ################################################################################################################

        # Go to final state
        for _ in range(12):
            _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        for _ in range(5):
            new_state, rewards, is_final, info = self.environment.step(action=self.environment.actions.get('UP'))

        self.assertEqual((19, 0), new_state)
        self.assertEqual([10, 0], rewards)
        self.assertTrue(is_final)
