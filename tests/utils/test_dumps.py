"""
Unit tests path where testing dumps feature from agentMOMP with each environment
"""

import unittest

import numpy as np

from agents import AgentPQL
from environments import *
from models import Vector, EvaluationMechanism, GraphType


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
        initial_state = ((1, 1), False)
        default_reward = (1, 1)
        seed = 1
        hv_reference = Vector([-100, -100, -100])
        evaluation_mechanism = EvaluationMechanism.PO
        epsilon = 0.4
        states_to_observe = {((1, 1), False)}
        episodes = np.random.randint(1, 10)
        gamma = 0.99
        max_steps = None

        # Vector configuration
        Vector.set_decimal_precision(decimal_precision=2)

        # Instance of Environment
        env = BonusWorld(initial_state=initial_state, default_reward=default_reward, seed=seed)

        # Instance of AgentMOMP
        agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                         max_steps=max_steps)

        # Train to modify train_data.
        agent.episode_train(episodes=episodes, graph_type=GraphType.EPISODES)

        # Save and load as new agent.
        agent.save()
        agent_loaded = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_steps, agent_loaded.max_steps)
        self.assertEqual(agent.graph_info, agent_loaded.states_to_observe)
        self.assertEqual(agent.initial_seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.initial_state, agent_loaded.environment.initial_state)
        self.assertEqual(agent.environment.initial_seed, agent_loaded.environment.initial_seed)
        self.assertEqual(agent.environment.default_reward, agent_loaded.environment.default_reward)
        self.assertEqual(agent.environment.bonus_activated, agent_loaded.environment.bonus_activated)

    def test_buridan_ass(self):
        """
        Testing agent with BuridanAss environment.
        :return:
        """
        initial_state = (0, 0)
        default_reward = (1., 1., 1.)
        seed = 1
        hv_reference = Vector([-5, -5, -5])
        evaluation_mechanism = EvaluationMechanism.PO
        epsilon = 0.11
        states_to_observe = {(1, 1)}
        episodes = np.random.randint(10, 50)
        gamma = 0.99
        max_steps = None
        p_stolen = .8
        n_appear = 15
        stolen_penalty = -.3
        walking_penalty = -2
        hunger_penalty = -2
        last_ate_limit = 15

        # Instance of Environment
        env = BuridanAss(initial_state=initial_state, default_reward=default_reward, seed=seed,
                         p_stolen=p_stolen, n_appear=n_appear, stolen_penalty=stolen_penalty,
                         walking_penalty=walking_penalty, hunger_penalty=hunger_penalty,
                         last_ate_limit=last_ate_limit)

        # Instance of AgentMOMP
        agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                         max_steps=max_steps)

        # Train to modify train_data.
        agent.episode_train(episodes=episodes, graph_type=GraphType.EPISODES)

        # Save and load as new agent.
        agent.save()
        agent_loaded = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_steps, agent_loaded.max_steps)
        self.assertEqual(agent.graph_info, agent_loaded.states_to_observe)
        self.assertEqual(agent.initial_seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.initial_state, agent_loaded.environment.initial_state)
        self.assertEqual(agent.environment.initial_seed, agent_loaded.environment.initial_seed)
        self.assertEqual(agent.environment.default_reward, agent_loaded.environment.default_reward)
        self.assertEqual(agent.environment.p_stolen, agent_loaded.environment.p_stolen)
        self.assertEqual(agent.environment.n_appear, agent_loaded.environment.n_appear)
        self.assertEqual(agent.environment.stolen_penalty, agent_loaded.environment.stolen_penalty)
        self.assertEqual(agent.environment.walking_penalty, agent_loaded.environment.walking_penalty)
        self.assertEqual(agent.environment.hunger_penalty, agent_loaded.environment.hunger_penalty)
        self.assertEqual(agent.environment.last_ate_limit, agent_loaded.environment.last_ate_limit)

    def test_deep_sea_treasure(self):
        """
        Testing agent with DeepSeaTreasure environment.
        :return:
        """
        initial_state = (1, 1)
        default_reward = (1,)
        seed = 1
        hv_reference = Vector([-20, 0])
        evaluation_mechanism = EvaluationMechanism.PO
        epsilon = 0.4
        states_to_observe = {(0, 0)}
        episodes = np.random.randint(10, 50)
        gamma = 0.99
        max_steps = None

        # Instance of Environment
        env = DeepSeaTreasure(initial_state=initial_state, default_reward=default_reward, seed=seed)

        # Instance of AgentMOMP
        agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                         max_steps=max_steps)

        # Train to modify train_data.
        agent.episode_train(episodes=episodes, graph_type=GraphType.EPISODES)

        # Save and load as new agent.
        agent.save()
        agent_loaded = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_steps, agent_loaded.max_steps)
        self.assertEqual(agent.graph_info, agent_loaded.states_to_observe)
        self.assertEqual(agent.initial_seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.initial_state, agent_loaded.environment.initial_state)
        self.assertEqual(agent.environment.initial_seed, agent_loaded.environment.initial_seed)
        self.assertEqual(agent.environment.default_reward, agent_loaded.environment.default_reward)

    def test_deep_sea_treasure_stochastic(self):
        """
        Testing agent with DeepSeaTreasureTransitions environment.
        :return:
        """
        initial_state = (1, 1)
        default_reward = (1,)
        seed = 1
        hv_reference = Vector([-20, 0])
        evaluation_mechanism = EvaluationMechanism.PO
        epsilon = 0.11
        states_to_observe = {(0, 0)}
        episodes = np.random.randint(10, 50)
        gamma = 0.99
        max_steps = None
        n_transaction = 0.33

        # Instance of Environment
        env = DeepSeaTreasureStochastic(initial_state=initial_state, default_reward=default_reward, seed=seed,
                                        n_transition=n_transaction)

        # Instance of AgentMOMP
        agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                         max_steps=max_steps)

        # Train to modify train_data.
        agent.episode_train(episodes=episodes, graph_type=GraphType.EPISODES)

        # Save and load as new agent.
        agent.save()
        agent_loaded = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_steps, agent_loaded.max_steps)
        self.assertEqual(agent.graph_info, agent_loaded.states_to_observe)
        self.assertEqual(agent.initial_seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.initial_state, agent_loaded.environment.initial_state)
        self.assertEqual(agent.environment.initial_seed, agent_loaded.environment.initial_seed)
        self.assertEqual(agent.environment.default_reward, agent_loaded.environment.default_reward)
        self.assertEqual(agent.environment.transitions, agent_loaded.environment.transitions)

    def test_linked_rings(self):
        """
        Testing agent with LinkedRings environment.
        :return:
        """
        seed = 1
        hv_reference = Vector([-100, -100])
        evaluation_mechanism = EvaluationMechanism.PO
        epsilon = 0.11
        states_to_observe = {0}
        episodes = np.random.randint(10, 50)
        gamma = 0.99
        max_steps = 10
        initial_state = 1

        # Instance of Environment
        env = LinkedRings(seed=seed, initial_state=initial_state)

        # Instance of AgentMOMP
        agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                         max_steps=max_steps)

        # Train to modify train_data.
        agent.episode_train(episodes=episodes, graph_type=GraphType.EPISODES)

        # Save and load as new agent.
        agent.save()
        agent_loaded = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_steps, agent_loaded.max_steps)
        self.assertEqual(agent.graph_info, agent_loaded.states_to_observe)
        self.assertEqual(agent.initial_seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.initial_seed, agent_loaded.environment.initial_seed)
        self.assertEqual(agent.environment.initial_state, agent_loaded.environment.initial_state)

    def test_non_recurrent_rings(self):
        """
        Testing agent with NonRecurrentRings environment.
        :return:
        """
        seed = 1
        hv_reference = Vector([-100, -100])
        evaluation_mechanism = EvaluationMechanism.PO
        epsilon = 0.11
        states_to_observe = {0}
        episodes = np.random.randint(10, 50)
        gamma = 0.99
        max_steps = 10
        initial_state = 1

        # Instance of Environment
        env = NonRecurrentRings(seed=seed, initial_state=initial_state)

        # Instance of AgentMOMP
        agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                         max_steps=max_steps)

        # Train to modify train_data.
        agent.episode_train(episodes=episodes, graph_type=GraphType.EPISODES)

        # Save and load as new agent.
        agent.save()
        agent_loaded = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_steps, agent_loaded.max_steps)
        self.assertEqual(agent.graph_info, agent_loaded.states_to_observe)
        self.assertEqual(agent.initial_seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.initial_seed, agent_loaded.environment.initial_seed)
        self.assertEqual(agent.environment.initial_state, agent_loaded.environment.initial_state)

    def test_mo_puddle_world(self):
        """
        Testing agent with MoPuddleWorld environment.
        :return:
        """
        default_reward = (10, -1)
        penalize_non_goal = -1.001
        seed = 1
        hv_reference = Vector([-100, -100])
        evaluation_mechanism = EvaluationMechanism.PO
        epsilon = 0.3
        states_to_observe = {(0, 0)}
        episodes = np.random.randint(10, 40)
        gamma = 0.99
        max_steps = None

        # Instance of Environment
        env = MoPuddleWorld(default_reward=default_reward, seed=seed, penalize_non_goal=penalize_non_goal)

        # Instance of AgentMOMP
        agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                         max_steps=max_steps)

        # Train to modify train_data.
        agent.episode_train(episodes=episodes, graph_type=GraphType.EPISODES)

        # Save and load as new agent.
        agent.save()
        agent_loaded = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_steps, agent_loaded.max_steps)
        self.assertEqual(agent.graph_info, agent_loaded.states_to_observe)
        self.assertEqual(agent.initial_seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.penalize_non_goal, agent_loaded.environment.penalize_non_goal)
        self.assertEqual(agent.environment.initial_seed, agent_loaded.environment.initial_seed)
        self.assertEqual(agent.environment.default_reward, agent_loaded.environment.default_reward)

    def test_pressurized_bountiful_sea_treasure(self):
        """
        Testing agent with PressurizedBountifulSeaTreasure environment.
        :return:
        """
        default_reward = (-1,)
        seed = 1
        hv_reference = Vector([-20, -20, -20])
        evaluation_mechanism = EvaluationMechanism.PO
        epsilon = 0.11
        states_to_observe = {(0, 0)}
        episodes = np.random.randint(10, 50)
        gamma = 0.99
        max_steps = None

        # Instance of Environment
        env = PressurizedBountifulSeaTreasure(default_reward=default_reward, seed=seed)

        # Instance of AgentMOMP
        agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                         max_steps=max_steps)

        # Train to modify train_data.
        agent.episode_train(episodes=episodes, graph_type=GraphType.EPISODES)

        # Save and load as new agent.
        agent.save()
        agent_loaded = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_steps, agent_loaded.max_steps)
        self.assertEqual(agent.graph_info, agent_loaded.states_to_observe)
        self.assertEqual(agent.initial_seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.initial_seed, agent_loaded.environment.initial_seed)
        self.assertEqual(agent.environment.default_reward, agent_loaded.environment.default_reward)

    def test_resource_gathering(self):
        """
        Testing agent with ResourceGathering environment.
        :return:
        """
        default_reward = (-1, 0.1, 0.1)
        seed = 1
        hv_reference = Vector([-20, -20, -20])
        evaluation_mechanism = EvaluationMechanism.PO
        epsilon = 0.11
        states_to_observe = {(0, 0)}
        episodes = np.random.randint(10, 50)
        gamma = 0.99
        max_steps = None
        p_attack = 0.2

        # Instance of Environment
        env = ResourceGathering(default_reward=default_reward, seed=seed, p_attack=p_attack)

        # Instance of AgentMOMP
        agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                         max_steps=max_steps)

        # Train to modify train_data.
        agent.episode_train(episodes=episodes, graph_type=GraphType.EPISODES)

        # Save and load as new agent.
        agent.save()
        agent_loaded = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_steps, agent_loaded.max_steps)
        self.assertEqual(agent.graph_info, agent_loaded.states_to_observe)
        self.assertEqual(agent.initial_seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.p_attack, agent_loaded.environment.p_attack)
        self.assertEqual(agent.environment.initial_seed, agent_loaded.environment.initial_seed)
        self.assertEqual(agent.environment.default_reward, agent_loaded.environment.default_reward)

    def test_resource_gathering_limit(self):
        """
        Testing agent with ResourceGatheringLimit environment.
        :return:
        """
        default_reward = (-1, 0, 0)
        seed = 1
        time_limit = 50
        hv_reference = Vector([-20, -20, -20])
        evaluation_mechanism = EvaluationMechanism.PO
        epsilon = 0.4
        states_to_observe = {(0, 0)}
        episodes = np.random.randint(10, 50)
        gamma = 0.99
        max_steps = None
        p_attack = 0.2

        # Instance of Environment
        env = ResourceGatheringLimit(default_reward=default_reward, seed=seed, p_attack=p_attack,
                                     time_limit=time_limit)

        # Instance of AgentMOMP
        agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                         max_steps=max_steps)

        # Train to modify train_data.
        agent.episode_train(episodes=episodes, graph_type=GraphType.EPISODES)

        # Save and load as new agent.
        agent.save()
        agent_loaded = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_steps, agent_loaded.max_steps)
        self.assertEqual(agent.graph_info, agent_loaded.states_to_observe)
        self.assertEqual(agent.initial_seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.p_attack, agent_loaded.environment.p_attack)
        self.assertEqual(agent.environment.time_limit, agent_loaded.environment.time_limit)
        self.assertEqual(agent.environment.initial_seed, agent_loaded.environment.initial_seed)
        self.assertEqual(agent.environment.default_reward, agent_loaded.environment.default_reward)

    def test_space_exploration(self):
        """
        Testing agent with SpaceExploration environment.
        :return:
        """
        default_reward = (1, 0)
        seed = 1
        hv_reference = Vector([-20, -20])
        evaluation_mechanism = EvaluationMechanism.PO
        epsilon = 0.11
        states_to_observe = {(0, 0)}
        episodes = np.random.randint(10, 50)
        gamma = 0.99
        max_steps = None

        # Instance of Environment
        env = SpaceExploration(default_reward=default_reward, seed=seed)

        # Instance of AgentMOMP
        agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                         hv_reference=hv_reference, evaluation_mechanism=evaluation_mechanism, gamma=gamma,
                         max_steps=max_steps)

        # Train to modify train_data.
        agent.episode_train(episodes=episodes, graph_type=GraphType.EPISODES)

        # Save and load as new agent.
        agent.save()
        agent_loaded = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

        # Agent
        self.assertEqual(type(agent), type(agent_loaded))
        self.assertEqual(agent.epsilon, agent_loaded.epsilon)
        self.assertEqual(agent.gamma, agent_loaded.gamma)
        self.assertEqual(agent.max_steps, agent_loaded.max_steps)
        self.assertEqual(agent.graph_info, agent_loaded.states_to_observe)
        self.assertEqual(agent.initial_seed, agent_loaded.seed)
        self.assertEqual(agent.r, agent_loaded.r)
        self.assertEqual(agent.nd, agent_loaded.nd)
        self.assertEqual(agent.n, agent_loaded.n)
        self.assertEqual(agent.hv_reference, agent_loaded.hv_reference)
        self.assertEqual(agent.evaluation_mechanism, agent_loaded.evaluation_mechanism)

        # Environment
        self.assertEqual(agent.environment.initial_seed, agent_loaded.environment.initial_seed)
        self.assertEqual(agent.environment.default_reward, agent_loaded.environment.default_reward)
