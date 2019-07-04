"""
Unit tests file where testing Algorithm 1 agent.
"""

import unittest

import numpy as np

from agents import AgentA1
from gym_tiadas.gym_tiadas.envs import DeepSeaTreasureRightDown
from models import Vector


class TestA1(unittest.TestCase):
    environment = DeepSeaTreasureRightDown()

    def setUp(self):
        # Set seed to 0 to testing.
        self.seed = 0

        # Build instance of agent
        self.agent = AgentA1(seed=self.seed, environment=self.environment, integer_mode=True,
                             hv_reference=Vector([-25, 0]))

    def tearDown(self):
        self.agent = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # This agent must have next attributes
        self.assertTrue(hasattr(self.agent, 'q') and isinstance(self.agent.q, dict))
        self.assertTrue(hasattr(self.agent, 's') and isinstance(self.agent.s, dict))
        self.assertTrue(hasattr(self.agent, 'v') and isinstance(self.agent.v, dict))
        self.assertTrue(hasattr(self.agent, 'indexes_counter') and isinstance(self.agent.indexes_counter, dict))
        self.assertTrue(hasattr(self.agent, 'integer_mode') and isinstance(self.agent.integer_mode, bool))
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
