"""
Unit tests file where testing test ResourceGatheringLimit environment.
"""
import random

from environments import ResourceGatheringEpisodic
from tests.environments.test_resource_gathering import TestResourceGathering


class TestResourceGatheringLimit(TestResourceGathering):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = ResourceGatheringEpisodic(seed=0)

    def test_init(self):

        # Super method call
        super().test_init()

        self.assertTrue(hasattr(self.environment, 'steps'))
        self.assertTrue(hasattr(self.environment, 'steps_limit'))

    def test_reset(self):

        self.environment.steps = random.randint(0, 1000)

        # Super method call
        super().test_reset()

        self.assertEqual(0, self.environment.steps)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Simple valid step
        # Reward:
        #   [enemy_attack, gold, gems]
        # Complex position:
        #   (position, resources_available)
        # Remember that initial position is (2, 4)

        # Disable enemy attack
        self.environment.p_attack = 0

        next_state, reward, is_final = None, None, None

        # Do 2 steps to RIGHT
        for _ in range(2):
            _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # Do 3 steps to UP (Get a gem)
        for _ in range(3):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual(((4, 1), (0, 1)), next_state)
        self.assertEqual([0, 0, 0], reward)
        self.assertFalse(is_final)

        _ = self.environment.step(action=self.environment.actions['UP'])

        # Do 2 steps to LEFT
        for _ in range(2):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['LEFT'])

        self.assertEqual(((2, 0), (1, 1)), next_state)
        self.assertEqual([0, 0, 0], reward)
        self.assertFalse(is_final)

        # Go to home
        # Do 4 steps to DOWN
        for _ in range(4):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual(((2, 4), (1, 1)), next_state)
        self.assertEqual([0, 1, 1], reward)
        self.assertTrue(is_final)

        ################################################################################################################
        # Trying get gold through enemy
        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Do 4 steps to UP
        for _ in range(4):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual(((2, 0), (1, 0)), next_state)
        self.assertEqual([0, 0, 0], reward)
        self.assertFalse(is_final)

        # Force to enemy attack
        self.environment.p_attack = 1

        # Go to enemy position
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        # Reset at home
        self.assertEqual(((2, 4), (0, 0)), next_state)
        self.assertEqual([-1, 0, 0], reward)
        self.assertTrue(is_final)

        # Set a state with gold and gem
        self.environment.current_state = ((3, 4), (1, 1))

        # Waste time
        steps_used = self.environment.steps

        # Do steps until time_limit
        for _ in range(self.environment.steps_limit - steps_used):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        self.assertEqual(((4, 4), (1, 1)), next_state)
        self.assertEqual([0, 0, 0], reward)
        self.assertTrue(is_final)

    def test_states_size(self):
        self.assertEqual(93, len(self.environment.states()))