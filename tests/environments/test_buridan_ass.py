"""
Unit tests file where testing BuridanAss environment.
"""

import unittest

from gym import spaces

from environments import BuridanAss


class TestBuridanAss(unittest.TestCase):
    environment = None

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = BuridanAss(seed=0)

    def tearDown(self):
        self.environment = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # This environment must have another attributes
        self.assertTrue(hasattr(self.environment, 'p_stolen'))
        self.assertTrue(hasattr(self.environment, 'n_appear'))
        self.assertTrue(hasattr(self.environment, 'walking_penalty'))
        self.assertTrue(hasattr(self.environment, 'stolen_penalty'))
        self.assertTrue(hasattr(self.environment, 'hunger_penalty'))
        self.assertTrue(hasattr(self.environment, 'last_ate_limit'))
        self.assertTrue(hasattr(self.environment, 'last_ate'))

        # By default mesh shape is 3x3
        self.assertEqual(spaces.Tuple((spaces.Discrete(3), spaces.Discrete(3))), self.environment.observation_space)

        # By default action space is 5 (UP, RIGHT, DOWN, LEFT, STAY)
        self.assertIsInstance(self.environment.action_space, spaces.Space)

        # By default initial state is (1, 1)
        self.assertEqual((1, 1), self.environment.initial_state)
        self.assertEqual(self.environment.initial_state, self.environment.current_state)

        # Default reward is (0., 0., 0.)
        self.assertEqual((0., 0., 0.), self.environment.default_reward)

        # Check if finals states are correct
        for state, food in self.environment.finals.items():
            self.assertTrue(self.environment.observation_space.contains(state))
            self.assertTrue(food)

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
        # Set last ate randomly
        self.environment.last_ate = self.environment.np_random.randint(0, self.environment.last_ate_limit)
        # Set all food state to False
        for state in self.environment.finals.keys():
            self.environment.finals.update({state: False})

        # Reset environment
        self.environment.reset()

        # Asserts
        self.assertEqual(self.environment.initial_state, self.environment.current_state)
        self.assertEqual(0, self.environment.last_ate)

        # Check if finals states are correct
        for state, food in self.environment.finals.items():
            self.assertTrue(self.environment.observation_space.contains(state))
            self.assertTrue(food)

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
        new_state = self.environment.next_state(action=self.environment.actions.get('UP'))
        self.assertEqual(state, new_state)

        # Go to RIGHT (increment x axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual((state[0] + 1, state[1]), new_state)

        # Go to DOWN (increment y axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual((state[0], state[1] + 1), new_state)

        # Cannot go to LEFT (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual(state, new_state)

        ################################################################################################################
        # Set to (2, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (2, 0)
        self.environment.current_state = state

        # Cannot go to UP (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('UP'))
        self.assertEqual(state, new_state)

        # Cannot go to RIGHT (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual(state, new_state)

        # Go to DOWN (increment y axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual((state[0], state[1] + 1), new_state)

        # Go to LEFT (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual((state[0] - 1, state[1]), new_state)

        ################################################################################################################
        # Set to (2, 2) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (2, 2)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('UP'))
        self.assertEqual((state[0], state[1] - 1), new_state)

        # Cannot go to RIGHT (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual(state, new_state)

        # Cannot go to DOWN (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual(state, new_state)

        # Go to LEFT (decrement x axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual((state[0] - 1, state[1]), new_state)

        ################################################################################################################
        # Set to (0, 2) (DOWN-LEFT corner)
        ################################################################################################################
        state = (0, 2)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('UP'))
        self.assertEqual((state[0], state[1] - 1), new_state)

        # Go to RIGHT (increment x axis)
        new_state = self.environment.next_state(action=self.environment.actions.get('RIGHT'))
        self.assertEqual((state[0] + 1, state[1]), new_state)

        # Cannot go to DOWN (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('DOWN'))
        self.assertEqual(state, new_state)

        # Cannot go to LEFT (Keep in same state
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual(state, new_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Simple valid step, at each step penalizes -1.
        new_state, rewards, is_final, info = self.environment.step(action=self.environment.actions.get('DOWN'))

        # State:
        #   (state, states_visible_with_food, last_ate)
        # Reward:
        #   [hungry_penalize, stolen_penalize, step_penalize]

        # Remember that initial state is (1, 1), and this problem return a complex state
        self.assertEqual(((1, 2), (2, 2), 1), new_state)
        self.assertEqual([0, 0, -1], rewards)
        self.assertFalse(is_final)
        self.assertFalse(info)

        # Return at begin, to see both food stacks.
        new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('UP'))

        self.assertEqual(((1, 1), ((0, 0), (2, 2)), 2), new_state)
        self.assertEqual([0, 0, -1], rewards)
        self.assertFalse(is_final)

        # Go to RIGHT
        new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        self.assertEqual(((2, 1), (2, 2), 3), new_state)
        # Not visible food stack (0, 0) is stolen
        self.assertEqual([0, -0.5, -1], rewards)
        self.assertFalse(is_final)

        # Go to DOWN
        new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        self.assertEqual(((2, 2), (2, 2), 4), new_state)
        self.assertEqual([0, 0, -1], rewards)
        self.assertFalse(is_final)

        # Go to STAY
        new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('STAY'))

        # Not more food stacks, donkey has ate.
        self.assertEqual(((2, 2), (), 0), new_state)
        self.assertEqual([0, 0, 0], rewards)
        # Not more food
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Wasteful steps for the donkey to be hungry
        for _ in range(10):
            new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        # Donkey has hungry.
        self.assertEqual(((2, 1), (2, 2), 9), new_state)
        # Hungry penalize active and (0, 0) food stack stolen
        self.assertEqual([-1.0, -0.5, -1.0], rewards)
        self.assertFalse(is_final)

        # Go to DOWN (2, 2)
        new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('DOWN'))

        # Go to STAY (2, 2)
        new_state, rewards, is_final, _ = self.environment.step(action=self.environment.actions.get('STAY'))

        # Donkey has ate.
        self.assertEqual(((2, 2), (), 0), new_state)
        self.assertEqual([0, 0, 0], rewards)
        self.assertTrue(is_final)
