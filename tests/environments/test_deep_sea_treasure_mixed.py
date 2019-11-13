"""
Unit tests file where testing DeepSeaTreasure environment.
"""

from environments import DeepSeaTreasureMixed
from models import Vector
from tests.environments.test_deep_sea_treasure import TestDeepSeaTreasure


class TestDeepSeaTreasureMixed(TestDeepSeaTreasure):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = DeepSeaTreasureMixed(seed=0)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Simple valid step
        # Reward:
        #   [time_inverted, treasure_value]
        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions.get('DOWN'))

        # Remember that initial position is (0, 0)
        self.assertEqual((0, 1), next_state)
        self.assertEqual([-1, 1], reward)
        self.assertTrue(is_final)
        self.assertFalse(info)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final position.
        # Go to right 6 steps, until (6, 0).
        for _ in range(6):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        # Go to down 6 steps, until (6, 6).
        for _ in range(6):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        # Try to go LEFT, but is an obstacle.
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions.get('LEFT'))

        self.assertEqual((6, 6), next_state)
        self.assertEqual([-1, 0], reward)
        self.assertFalse(is_final)

        # Go to DOWN to get 24 treasure-value.
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        self.assertEqual((6, 7), next_state)
        self.assertEqual([-1, 15], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final position.
        # Go to right 9 steps, until (9, 0).
        for _ in range(9):
            _, _, _, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        # Go to down 10 steps, until (9, 10).
        for _ in range(10):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        self.assertEqual((9, 10), next_state)
        self.assertEqual([-1, 20], reward)
        self.assertTrue(is_final)

    def test_transition_reward(self):

        # In this environment doesn't mind initial state to get the reward
        state = self.environment.observation_space.sample()

        # Doesn't mind action too.
        action = self.environment.action_space.sample()

        # An intermediate state
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(1, 1)
            ), self.environment.default_reward
        )

        # A final state
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(1, 2)
            ), Vector([-1, 2])
        )
