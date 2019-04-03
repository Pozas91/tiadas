"""
Unit tests file where testing SpaceExploration environment.
"""

import unittest

from gym import spaces

from gym_tiadas.gym_tiadas.envs import SpaceExploration


class TestSpaceExploration(unittest.TestCase):
    environment = None

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = SpaceExploration(seed=0)

    def tearDown(self):
        self.environment = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # All environments must be have an action_space and an observation_space attributes.
        self.assertTrue(hasattr(self.environment, 'action_space'))
        self.assertTrue(hasattr(self.environment, 'observation_space'))

        # All environments must be have an step, seed, reset, render and _next_state methods.
        self.assertTrue(hasattr(self.environment, 'step'))
        self.assertTrue(hasattr(self.environment, 'seed'))
        self.assertTrue(hasattr(self.environment, 'reset'))
        self.assertTrue(hasattr(self.environment, '_next_state'))

        # This environment must have another attributes
        self.assertTrue(hasattr(self.environment, 'finals'))
        self.assertTrue(hasattr(self.environment, 'obstacles'))
        self.assertTrue(hasattr(self.environment, 'radiations'))
        self.assertTrue(hasattr(self.environment, 'asteroids'))

        # By default mesh shape is 13x5
        self.assertEqual(spaces.Tuple((spaces.Discrete(13), spaces.Discrete(5))), self.environment.observation_space)

        # By default action space is 8 (UP, UP-RIGHT, RIGHT, DOWN-RIGHT, DOWN, DOWN-LEFT, LEFT, UP-LEFT)
        self.assertEqual(spaces.Discrete(8), self.environment.action_space)

        # By default initial state is (5, 2)
        self.assertEqual((5, 2), self.environment.initial_state)
        self.assertEqual(self.environment.initial_state, self.environment.current_state)

        # Default reward is 0.
        self.assertEqual(0., self.environment.default_reward)

    def test_seed(self):
        """
        Testing seed method
        :return:
        """
        self.environment.seed(seed=0)
        n1_1 = self.environment.np_random.randint(0, 10)
        n1_2 = self.environment.np_random.randint(0, 10)

        self.environment.seed(seed=0)
        n2_1 = self.environment.np_random.randint(0, 10)
        n2_2 = self.environment.np_random.randint(0, 10)

        self.assertEqual(n1_1, n2_1)
        self.assertEqual(n1_2, n2_2)

    def test_reset(self):
        """
        Testing reset method
        :return:
        """

        # Set current state to random state
        self.environment.current_state = self.environment.observation_space.sample()

        # Reset environment
        self.environment.reset()

        # Asserts
        self.assertEqual(self.environment.initial_state, self.environment.current_state)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        # THIS IS A CYCLIC ENVIRONMENT

        ################################################################################################################
        # Begin at state (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        state = (0, 0)
        self.environment.current_state = state

        # Go to UP
        new_state = self.environment._next_state(action=self.environment.actions.get('UP'))
        self.assertEqual((0, 4), new_state)

        # Go to UP RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('UP RIGHT'))
        self.assertEqual((1, 4), new_state)

        # Go to RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual((1, 0), new_state)

        # Go to DOWN RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN RIGHT'))
        self.assertEqual((1, 1), new_state)

        # Go to DOWN
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual((0, 1), new_state)

        # Go to DOWN LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN LEFT'))
        self.assertEqual((12, 1), new_state)

        # Go to LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual((12, 0), new_state)

        # Go to UP LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('UP LEFT'))
        self.assertEqual((12, 4), new_state)

        ################################################################################################################
        # Begin at state (12, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (12, 0)
        self.environment.current_state = state

        # Go to UP
        new_state = self.environment._next_state(action=self.environment.actions.get('UP'))
        self.assertEqual((12, 4), new_state)

        # Go to UP RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('UP RIGHT'))
        self.assertEqual((0, 4), new_state)

        # Go to RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual((0, 0), new_state)

        # Go to DOWN RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN RIGHT'))
        self.assertEqual((0, 1), new_state)

        # Go to DOWN
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual((12, 1), new_state)

        # Go to DOWN LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN LEFT'))
        self.assertEqual((11, 1), new_state)

        # Go to LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual((11, 0), new_state)

        # Go to UP LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('UP LEFT'))
        self.assertEqual((11, 4), new_state)

        ################################################################################################################
        # Begin at state (12, 4) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (12, 4)
        self.environment.current_state = state

        # Go to UP
        new_state = self.environment._next_state(action=self.environment.actions.get('UP'))
        self.assertEqual((12, 3), new_state)

        # Go to UP RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('UP RIGHT'))
        self.assertEqual((0, 3), new_state)

        # Go to RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual((0, 4), new_state)

        # Go to DOWN RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN RIGHT'))
        self.assertEqual((0, 0), new_state)

        # Go to DOWN
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual((12, 0), new_state)

        # Go to DOWN LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN LEFT'))
        self.assertEqual((11, 0), new_state)

        # Go to LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual((11, 4), new_state)

        # Go to UP LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('UP LEFT'))
        self.assertEqual((11, 3), new_state)

        ################################################################################################################
        # Begin at state (0, 4) (DOWN-LEFT corner)
        ################################################################################################################
        state = (0, 4)
        self.environment.current_state = state

        # Go to UP
        new_state = self.environment._next_state(action=self.environment.actions.get('UP'))
        self.assertEqual((0, 3), new_state)

        # Go to UP RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('UP RIGHT'))
        self.assertEqual((1, 3), new_state)

        # Go to RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual((1, 4), new_state)

        # Go to DOWN RIGHT
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN RIGHT'))
        self.assertEqual((1, 0), new_state)

        # Go to DOWN
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual((0, 0), new_state)

        # Go to DOWN LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN LEFT'))
        self.assertEqual((12, 0), new_state)

        # Go to LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual((12, 4), new_state)

        # Go to UP LEFT
        new_state = self.environment._next_state(action=self.environment.actions.get('UP LEFT'))
        self.assertEqual((12, 3), new_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Reward:
        #   [mission_success, radiation]
        # Remember that initial state is (5, 2)

        # Reset environment
        self.environment.reset()

        # Go to RIGHT 2 steps, until (5, 4), ship is destroyed by asteroid.
        for _ in range(2):
            new_state, reward, is_final, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        self.assertEqual((7, 2), new_state)
        self.assertEqual([-100, -1], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to a final state.
        # (4, 2)
        _ = self.environment.step(action=self.environment.actions.get('LEFT'))

        # (3, 1)
        _ = self.environment.step(action=self.environment.actions.get('UP LEFT'))

        # (2, 1)
        _ = self.environment.step(action=self.environment.actions.get('LEFT'))

        # (1, 0), state with radiation
        new_state, reward, is_final, _ = self.environment.step(action=self.environment.actions.get('UP LEFT'))

        self.assertEqual((1, 0), new_state)
        self.assertEqual([0, -11], reward)
        self.assertFalse(is_final)

        # (0, 4)
        new_state, reward, is_final, _ = self.environment.step(action=self.environment.actions.get('UP LEFT'))

        self.assertEqual((0, 4), new_state)
        self.assertEqual([20, -1], reward)
        self.assertTrue(is_final)
