"""
Unit tests path where testing Base Agent.
"""

import unittest

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
        self.assertTrue(hasattr(self.agent, 'graph_info'))
        self.assertTrue(hasattr(self.agent, 'state'))
        self.assertTrue(hasattr(self.agent, 'initial_seed'))
        self.assertTrue(hasattr(self.agent, 'generator'))

        # All agents must be have next methods.
        self.assertTrue(hasattr(self.agent, 'show_graph_info'))
        self.assertTrue(hasattr(self.agent, 'print_information'))
        self.assertTrue(hasattr(self.agent, 'reset'))

    def test_reset_graph_info(self):
        """
        Testing reset graph extra method.
        :return:
        """

        # Define states to observe
        self.agent.states_to_observe = {(0, 0), (1, 1)}

        # Set graph extra
        self.agent.graph_info = {
            (0, 0): [1, 2, 3, 4, 5, 6],
            (1, 1): [1, 2, 30, 4, 5, 6],
        }

        # Reset graph extra
        self.agent.reset_graph_info()

        for state in self.agent.graph_info.keys():
            self.assertEqual(list, self.agent.graph_info.get(state))
