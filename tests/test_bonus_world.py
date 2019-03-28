"""
Unit tests file where testing bonus_world environment.
"""

import unittest

from gym import spaces

from gym_tiadas.gym_tiadas.envs import BonusWorld


class TestBonusWorld(unittest.TestCase):
    environment = None

    def setUp(self):
        self.environment = BonusWorld()

    def tearDown(self):
        self.environment = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # All environments must be have an action_space and an observation_space.
        self.assertTrue(hasattr(self.environment, 'action_space'))
        self.assertTrue(hasattr(self.environment, 'observation_space'))

        # By default mesh shape is 9x9
        self.assertEqual(spaces.Tuple((spaces.Discrete(9), spaces.Discrete(9))), self.environment.observation_space)

        # By default action space is 4 (UP, RIGHT, DOWN, LEFT)
        self.assertEqual(spaces.Discrete(4), self.environment.action_space)

        # By default initial state is (0, 0)
        self.assertEqual((0, 0), self.environment.initial_state)
        self.assertEqual(self.environment.initial_state, self.environment.current_state)
