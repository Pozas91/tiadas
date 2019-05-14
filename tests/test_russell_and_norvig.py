"""
Unit tests file where testing RusselAndNorvig environment.
"""

import unittest

from gym import spaces

from gym_tiadas.gym_tiadas.envs import RussellNorvig


class TestRusselAndNorvig(unittest.TestCase):
    environment = None

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = RussellNorvig(seed=0)

    def tearDown(self):
        self.environment = None

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # This environment must have another attributes
        self.assertTrue(hasattr(self.environment, 'transactions'))

        # By default mesh shape is 4x3
        self.assertEqual(spaces.Tuple((spaces.Discrete(4), spaces.Discrete(3))), self.environment.observation_space)

        # By default action space is 4 (UP, RIGHT, DOWN, LEFT)
        self.assertEqual(spaces.Discrete(4), self.environment.action_space)

        # By default initial state is (0, 2)
        self.assertEqual((0, 2), self.environment.initial_state)
        self.assertEqual(self.environment.initial_state, self.environment.current_state)

        # Default reward is (-0.04)
        self.assertEqual((-0.04,), self.environment.default_reward)

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
        # Set to (3, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (3, 0)
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
        # Set to (3, 2) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (3, 2)
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

        # Cannot go to LEFT (Keep in same state)
        new_state = self.environment.next_state(action=self.environment.actions.get('LEFT'))
        self.assertEqual(state, new_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Reward:
        #   [time_inverted, treasure_value]
        # Remember that initial state is (0, 2)

        # To testing force to go always in desired direction.
        self.environment.transactions = [1, 0, 0, 0]

        # Reset environment
        self.environment.reset()

        # Go to RIGHT 3 steps, until (3, 2).
        for _ in range(3):
            _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        new_state, reward, is_final, _ = self.environment.step(action=self.environment.actions.get('UP'))

        self.assertEqual((3, 1), new_state)
        self.assertEqual([-1], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final state.
        # Go to UP 9 steps, until (0, 0).
        for _ in range(2):
            _ = self.environment.step(action=self.environment.actions.get('UP'))

        # Go to RIGHT 3 steps, until (3, 0).
        for _ in range(3):
            new_state, reward, is_final, _ = self.environment.step(action=self.environment.actions.get('RIGHT'))

        self.assertEqual((3, 0), new_state)
        self.assertEqual([1], reward)
        self.assertTrue(is_final)

        ################################################################################################################
        # Testing transactions
        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Force DIR_90 transaction
        self.environment.transactions = [0, 1, 0, 0]

        # Agent turns to RIGHT
        new_state, reward, is_final, _ = self.environment.step(action=self.environment.actions.get('UP'))

        self.assertEqual((1, 2), new_state)
