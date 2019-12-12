"""
Unit tests file where testing EnvMesh.
"""

from environments.env_mesh import EnvMesh
from models import Vector
from tests.environments.test_environment import TestEnvironment


class TestEnvMesh(TestEnvironment):

    def setUp(self):
        # Mesh shape
        mesh_shape = (7, 7)

        # Default reward
        default_reward = Vector([1, 2, 1])

        # Obstacles
        obstacles = frozenset({
            (0, 0), (1, 1)
        })

        # Set initial_seed to 0 to testing.
        self.environment = EnvMesh(mesh_shape=mesh_shape, default_reward=default_reward, seed=0, obstacles=obstacles)

    def test_states(self):
        """
        Testing that all states must be contained into observation space
        :return:
        """
        self.assertTrue(
            all(
                self.environment.observation_space.contains(state) and
                state not in self.environment.obstacles and
                state not in self.environment.finals
                
                for state in self.environment.states()
            )
        )
