"""
Unit tests file where testing EnvMesh.
"""

import unittest

import gym

from gym_tiadas.gym_tiadas.envs.env_mesh import EnvMesh
from models import Vector


class TestEnvMesh(unittest.TestCase):

    def setUp(self):
        # Mesh shape
        mesh_shape = (7, 7)

        # Default reward
        default_reward = Vector([1, 2, 1])

        # Set seed to 0 to testing.
        self.environment = EnvMesh(mesh_shape=mesh_shape, default_reward=default_reward, seed=0)

    def tearDown(self):
        self.environment = None

    def test_observation_space(self):
        """"
        Testing observation_space attribute
        """

        self.assertIsInstance(self.environment.observation_space, gym.spaces.Tuple)
