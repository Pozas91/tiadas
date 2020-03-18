"""
Unit tests path where testing Base Agent.
"""

import unittest

import numpy as np

from agents import Agent
from environments.deep_sea_treasure import DeepSeaTreasure


class TestAgent(unittest.TestCase):
    environment = DeepSeaTreasure()

    def setUp(self):
        # Set initial_seed to 0 to testing.
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
        self.assertTrue(hasattr(self.agent, 'environment'))
        self.assertTrue(hasattr(self.agent, 'max_steps'))
        self.assertTrue(hasattr(self.agent, 'steps'))
        self.assertTrue(hasattr(self.agent, 'total_steps'))
        self.assertTrue(hasattr(self.agent, 'total_episodes'))
        self.assertTrue(hasattr(self.agent, 'graph_info'))
        self.assertTrue(hasattr(self.agent, 'position'))
        self.assertTrue(hasattr(self.agent, 'initial_seed'))
        self.assertTrue(hasattr(self.agent, 'generator'))

        # All agents must be have next methods.
        self.assertTrue(hasattr(self.agent, 'get_dict_model'))
        self.assertTrue(hasattr(self.agent, 'reset_steps'))
        self.assertTrue(hasattr(self.agent, 'show_graph_info'))
        self.assertTrue(hasattr(self.agent, 'print_information'))
        self.assertTrue(hasattr(self.agent, 'episode_train'))
        self.assertTrue(hasattr(self.agent, 'episode'))
        self.assertTrue(hasattr(self.agent, 'select_action'))
        self.assertTrue(hasattr(self.agent, 'reset'))
        self.assertTrue(hasattr(self.agent, '_best_action'))

    def test_reset_steps(self):
        """
        Testing reset steps method.
        :return:
        """

        # Set any steps
        self.agent.steps = np.random.randint(10, 1000)

        # Reset steps
        self.agent.reset_steps()

        self.assertEqual(self.agent.steps, 0)

    def test_reset_graph_info(self):
        """
        Testing reset graph extra method.
        :return:
        """

        # Set graph extra
        self.agent.graph_info = {
            (0, 0): [1, 2, 3, 4, 5, 6],
            (1, 1): [1, 2, 30, 4, 5, 6],
        }

        # Reset graph extra
        self.agent.reset_graph_info()

        for state in self.agent.graph_info.keys():
            self.assertEqual(list, self.agent.graph_info.get(state))
