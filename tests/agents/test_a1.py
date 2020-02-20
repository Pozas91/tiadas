"""
Unit tests path where testing Algorithm 1 agent.
"""

import unittest

import numpy as np

from agents import AgentA1
from environments import DeepSeaTreasureRightDown
from models import Vector


class TestA1(unittest.TestCase):
    environment = DeepSeaTreasureRightDown()

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.seed = 0

        # Build instance of agent
        self.agent = AgentA1(seed=self.seed, environment=self.environment, hv_reference=Vector([-25, 0]))

    def tearDown(self):
        self.agent = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # This agent must have next attributes
        self.assertTrue(hasattr(self.agent, 'q') and isinstance(self.agent.q, dict))
        self.assertTrue(hasattr(self.agent, 'state') and isinstance(self.agent.s, dict))
        self.assertTrue(hasattr(self.agent, 'v') and isinstance(self.agent.v, dict))
        self.assertTrue(hasattr(self.agent, 'indexes_counter') and isinstance(self.agent.indexes_counter, dict))
        self.assertTrue(hasattr(self.agent, 'hv_reference') and isinstance(self.agent.hv_reference, Vector))

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
