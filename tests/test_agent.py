"""
Unit tests file where testing Base Agent.
"""

import unittest

import numpy as np

from agents import Agent
from gym_tiadas.gym_tiadas.envs.deep_sea_treasure import DeepSeaTreasure


class TestAgent(unittest.TestCase):
    environment = DeepSeaTreasure()

    def setUp(self):
        # Set seed to 0 to testing.
        self.agent = Agent(seed=0, environment=self.environment)

    def tearDown(self):
        self.agent = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # All agents must be have next attributes
        self.assertTrue(hasattr(self.agent, 'gamma'))
        self.assertTrue(hasattr(self.agent, 'epsilon'))
        self.assertTrue(hasattr(self.agent, 'environment'))
        self.assertTrue(hasattr(self.agent, 'max_iterations'))
        self.assertTrue(hasattr(self.agent, 'iterations'))
        self.assertTrue(hasattr(self.agent, 'states_to_observe'))
        self.assertTrue(hasattr(self.agent, 'state'))
        self.assertTrue(hasattr(self.agent, 'seed'))
        self.assertTrue(hasattr(self.agent, 'generator'))

        # All agents must be have next methods.
        self.assertTrue(hasattr(self.agent, 'select_action'))
        self.assertTrue(hasattr(self.agent, 'episode'))
        self.assertTrue(hasattr(self.agent, 'reset'))
        self.assertTrue(hasattr(self.agent, 'best_action'))
        self.assertTrue(hasattr(self.agent, 'reset_iterations'))
        self.assertTrue(hasattr(self.agent, 'show_observed_states'))
        self.assertTrue(hasattr(self.agent, 'print_information'))
        self.assertTrue(hasattr(self.agent, 'train'))

    def test_reset_iterations(self):
        """
        Testing reset iterations method.
        :return:
        """

        # Set any iterations
        self.agent.iterations = np.random.randint(10, 1000)

        # Reset iterations
        self.agent.reset_iterations()

        self.assertEqual(self.agent.iterations, 0)

    def test_reset_states_to_observe(self):
        """
        Testing reset states to observe method.
        :return:
        """

        # Set states to observe
        self.agent.states_to_observe = {
            (0, 0): [1, 2, 3, 4, 5, 6],
            (1, 1): [1, 2, 30, 4, 5, 6],
        }

        # Reset states to observe
        self.agent.reset_states_to_observe()

        for state in self.agent.states_to_observe.keys():
            self.assertEqual(list, self.agent.states_to_observe.get(state))
