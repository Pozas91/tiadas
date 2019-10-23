"""
Unit tests file where testing Environment.
"""

import unittest

import gym

from environments import Environment
from models import Vector


class TestEnvironment(unittest.TestCase):

    def setUp(self):
        # An observation space
        observation_space = gym.spaces.Discrete(7)

        # Default reward
        default_reward = Vector([1, 2, 1])

        # Set seed to 0 to testing.
        self.environment = Environment(observation_space=observation_space, default_reward=default_reward, seed=0)

    def tearDown(self):
        self.environment = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # All agents must be have next attributes
        self.assertTrue(hasattr(self.environment, '_actions'))
        self.assertTrue(hasattr(self.environment, '_icons'))
        self.assertTrue(hasattr(self.environment, 'actions'))
        self.assertTrue(hasattr(self.environment, 'icons'))
        self.assertTrue(hasattr(self.environment, 'action_space'))
        self.assertTrue(hasattr(self.environment, 'observation_space'))
        self.assertTrue(hasattr(self.environment, 'np_random'))
        self.assertTrue(hasattr(self.environment, 'initial_seed'))
        self.assertTrue(hasattr(self.environment, 'initial_state'))
        self.assertTrue(hasattr(self.environment, 'current_state'))
        self.assertTrue(hasattr(self.environment, 'finals'))
        self.assertTrue(hasattr(self.environment, 'obstacles'))
        self.assertTrue(hasattr(self.environment, 'default_reward'))

        # All agents must be have next methods.
        self.assertTrue(hasattr(self.environment, 'step'))
        self.assertTrue(hasattr(self.environment, 'seed'))
        self.assertTrue(hasattr(self.environment, 'reset'))
        self.assertTrue(hasattr(self.environment, 'render'))
        self.assertTrue(hasattr(self.environment, 'next_state'))
        self.assertTrue(hasattr(self.environment, 'get_dict_model'))
        self.assertTrue(hasattr(self.environment, 'is_final'))

        self.assertIsInstance(self.environment.observation_space, gym.spaces.Space)
        self.assertIsInstance(self.environment.action_space, gym.spaces.Space)

        self.assertEqual(self.environment.initial_state, self.environment.current_state)

    def test_icons(self):
        """
        Testing icons property
        :return:
        """
        self.assertEqual(self.environment._icons, self.environment.icons)

    def test_actions(self):
        """
        Testing actions property
        :return:
        """
        self.assertEqual(self.environment._actions, self.environment.actions)

    def test_action_space_length(self):
        pass

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

        # Set current position to random position
        self.environment.current_state = self.environment.observation_space.sample()

        # Reset environment
        self.environment.reset()

        # Asserts
        self.assertEqual(self.environment.initial_state, self.environment.current_state)

    def test_get_dict_model(self):
        """"
        Testing get_dict_model method
        """

        model = self.environment.get_dict_model()

        self.assertEqual(self.environment.initial_seed, model.get('initial_seed'))
        self.assertEqual(self.environment.initial_state, model.get('initial_state'))
        self.assertEqual(self.environment.current_state, model.get('current_state'))
        self.assertEqual(self.environment.default_reward, model.get('default_reward'))

    def test_states(self):
        """
        Testing that all states must be contained into observation space
        :return:
        """
        pass

    def test_reachable_states(self):
        pass

    def test_transition_probability(self):
        pass

    def test_transition_reward(self):
        pass
