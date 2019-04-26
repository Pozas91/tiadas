"""
Unit tests file where testing test ResourceGatheringLimit environment.
"""

import unittest

from gym import spaces

from gym_tiadas.gym_tiadas.envs import ResourceGatheringLimit


class TestResourceGatheringLimit(unittest.TestCase):
    environment = None

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = ResourceGatheringLimit(seed=0)

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
        self.assertTrue(hasattr(self.environment, 'gold_states'))
        self.assertTrue(hasattr(self.environment, 'gem_states'))
        self.assertTrue(hasattr(self.environment, 'enemies'))
        self.assertTrue(hasattr(self.environment, 'obstacles'))

        # By default mesh shape is 5x5
        self.assertEqual(spaces.Tuple((spaces.Discrete(5), spaces.Discrete(5))), self.environment.observation_space)

        # By default action space is 4 (UP, RIGHT, DOWN, LEFT)
        self.assertEqual(spaces.Discrete(4), self.environment.action_space)

        # By default initial state is (2, 4)
        self.assertEqual((2, 4), self.environment.initial_state)
        self.assertEqual(self.environment.initial_state, self.environment.current_state)

        # Default reward is (0, 0, 0)
        self.assertEqual((0, 0, 0), self.environment.default_reward)

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
        self.environment.state[0] = self.environment.np_random.randint(-1, 0)
        self.environment.state[1] = self.environment.np_random.randint(0, 1)
        self.environment.state[2] = self.environment.np_random.randint(0, 1)

        # Get all golds
        for gold_state in self.environment.gold_states.keys():
            self.environment.gold_states.update({gold_state: False})

        # Get all gems
        for gem_state in self.environment.gem_states.keys():
            self.environment.gem_states.update({gem_state: False})

        # Reset environment
        self.environment.reset()

        # Asserts
        self.assertEqual(self.environment.initial_state, self.environment.current_state)
        self.assertEqual([0, 0, 0], self.environment.state)
        self.assertEqual(0, self.environment.time)

        for gold_state in self.environment.gold_states.values():
            self.assertTrue(gold_state)

        for gem_state in self.environment.gem_states.values():
            self.assertTrue(gem_state)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        ################################################################################################################
        # Begin at state (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        state = (0, 0)
        self.environment.current_state = state

        # Cannot go to UP (Keep in same state)
        new_state = self.environment._next_state(action=self.environment.actions.get('UP'))
        self.assertEqual(state, new_state)

        # Go to RIGHT (increment x axis)
        new_state = self.environment._next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual((state[0] + 1, state[1]), new_state)

        # Go to DOWN (increment y axis)
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual((state[0], state[1] + 1), new_state)

        # Cannot go to LEFT (Keep in same state)
        new_state = self.environment._next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual(state, new_state)

        ################################################################################################################
        # Set to (4, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (4, 0)
        self.environment.current_state = state

        # Cannot go to UP (Keep in same state)
        new_state = self.environment._next_state(action=self.environment.actions.get('UP'))
        self.assertEqual(state, new_state)

        # Cannot go to RIGHT (Keep in same state)
        new_state = self.environment._next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual(state, new_state)

        # Go to DOWN (increment y axis)
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual((state[0], state[1] + 1), new_state)

        # Go to LEFT (Keep in same state)
        new_state = self.environment._next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual((state[0] - 1, state[1]), new_state)

        ################################################################################################################
        # Set to (4, 4) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (4, 4)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        new_state = self.environment._next_state(action=self.environment.actions.get('UP'))
        self.assertEqual((state[0], state[1] - 1), new_state)

        # Cannot go to RIGHT (Keep in same state)
        new_state = self.environment._next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual(state, new_state)

        # Cannot go to DOWN (Keep in same state)
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual(state, new_state)

        # Go to LEFT (decrement x axis)
        new_state = self.environment._next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual((state[0] - 1, state[1]), new_state)

        ################################################################################################################
        # Set to (0, 4) (DOWN-LEFT corner)
        ################################################################################################################
        state = (0, 4)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        new_state = self.environment._next_state(action=self.environment.actions.get('UP'))
        self.assertEqual((state[0], state[1] - 1), new_state)

        # Go to RIGHT (increment x axis)
        new_state = self.environment._next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual((state[0] + 1, state[1]), new_state)

        # Cannot go to DOWN (Keep in same state)
        new_state = self.environment._next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual(state, new_state)

        # Cannot go to LEFT (Keep in same state)
        new_state = self.environment._next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual(state, new_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Simple valid step
        # Reward:
        #   [enemy_attack, gold, gems]
        # Complex state:
        #   (state, resources_available)
        # Remember that initial state is (2, 4)

        # Disable enemy attack
        self.environment.p_attack = 0

        # Do 2 steps to RIGHT
        for _ in range(2):
            _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        # Do 3 steps to UP (Get a gem)
        for _ in range(3):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('UP'))

        self.assertEqual(((4, 1), (0, 0, 1)), new_state)
        self.assertEqual([0, 0, 0], rewards)
        self.assertFalse(is_final)

        _ = self.environment.step(action=self.environment.actions.get('UP'))

        # Do 2 steps to LEFT
        for _ in range(2):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('LEFT'))

        self.assertEqual(((2, 0), (0, 1, 1)), new_state)
        self.assertEqual([0, 0, 0], rewards)
        self.assertFalse(is_final)

        # Go to home
        # Do 4 steps to DOWN
        for _ in range(4):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        self.assertEqual(((2, 4), (0, 1, 1)), new_state)
        self.assertEqual([0, 1, 1], rewards)
        self.assertTrue(is_final)

        ################################################################################################################
        # Trying get golf through enemy
        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Do 4 steps to UP
        for _ in range(4):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('UP'))

        self.assertEqual(((2, 0), (0, 1, 0)), new_state)
        self.assertEqual([0, 0, 0], rewards)
        self.assertFalse(is_final)

        # Force to enemy attack
        self.environment.p_attack = 1

        # Go to enemy position
        new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        # Reset at home
        self.assertEqual(((2, 4), (-1, 0, 0)), new_state)
        self.assertEqual([-1, 0, 0], rewards)
        self.assertTrue(is_final)

        ################################################################################################################
        # Now waste time.
        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Disable enemies attack
        self.environment.p_attack = 0

        # Do 4 steps to UP (to get gold)
        for _ in range(4):
            _ = self.environment.step(action=self.environment.actions.get('UP'))

        # Do 2 steps to RIGHT
        for _ in range(2):
            _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        # Get gem
        new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        # Now agent has gold and gem
        self.assertEqual(((4, 1), (0, 1, 1)), new_state)
        self.assertEqual([0, 0, 0], rewards)
        self.assertFalse(is_final)

        # Waste time

        time_used = self.environment.time

        # Do steps until time_limit
        for _ in range(self.environment.time_limit - time_used):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        self.assertEqual(((4, 1), (0, 1, 1)), new_state)
        self.assertEqual([0, 0.01, 0.01], rewards)
        self.assertTrue(is_final)
