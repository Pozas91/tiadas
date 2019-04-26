"""
Unit tests file where testing dumps feature from agentMOMP with each environment
"""

import unittest

import numpy as np

import utils.q_learning as uq
from gym_tiadas.gym_tiadas.envs import *
from models import AgentMOMP, Vector


class TestDumps(unittest.TestCase):

    def setUp(self) -> None:
        pass

    def tearDown(self) -> None:
        pass

    def test_bonus_world(self):
        """
        Testing agent with BonusWorld environment.
        :return:
        """
        initial_state = (1, 1)
        default_reward = (1, 1)
        seed = 1
        hv_reference = Vector([-5, -5, -5])
        evaluation_mechanism = 'PO-PQL'
        epsilon = 0.2
        states_to_observe = [(0, 0)]
        epochs = np.random.randint(10, 100)
        gamma = 0.8
        max_iterations = None

        environment = BonusWorld(initial_state=initial_state, default_reward=default_reward, seed=seed)

        agent = AgentMOMP(environment=environment, epsilon=epsilon, states_to_observe=states_to_observe,
                          hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                          max_iterations=max_iterations)

        uq.train(agent=agent, epochs=epochs)

        agent.save()
        agent_loaded = AgentMOMP.load()

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_iterations, agent_loaded.max_iterations)
        self.assertEqual(agent.states_to_observe, agent_loaded.states_to_observe)
        self.assertEqual(agent.seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.initial_state, agent_loaded.environment.initial_state)
        self.assertEqual(agent.environment.seed, agent_loaded.environment.seed)
        self.assertEqual(agent.environment.default_reward, agent_loaded.environment.default_reward)
        self.assertEqual(agent.environment.bonus_activated, agent_loaded.environment.bonus_activated)

    def test_buridan_ass(self):
        """
        Testing agent with BuridanAss environment.
        :return:
        """
        initial_state = (1, 1)
        default_reward = (1, 1)
        seed = 1
        hv_reference = Vector([-5, -5, -5])
        evaluation_mechanism = 'PO-PQL'
        epsilon = 0.2
        states_to_observe = [(0, 0)]
        epochs = np.random.randint(10, 100)
        gamma = 0.8
        max_iterations = None

        environment = BonusWorld(initial_state=initial_state, default_reward=default_reward, seed=seed)

        agent = AgentMOMP(environment=environment, epsilon=epsilon, states_to_observe=states_to_observe,
                          hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                          max_iterations=max_iterations)

        uq.train(agent=agent, epochs=epochs)

        agent.save()
        agent_loaded = AgentMOMP.load()

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_iterations, agent_loaded.max_iterations)
        self.assertEqual(agent.states_to_observe, agent_loaded.states_to_observe)
        self.assertEqual(agent.seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.initial_state, agent_loaded.environment.initial_state)
        self.assertEqual(agent.environment.seed, agent_loaded.environment.seed)
        self.assertEqual(agent.environment.default_reward, agent_loaded.environment.default_reward)
        self.assertEqual(agent.environment.bonus_activated, agent_loaded.environment.bonus_activated)
