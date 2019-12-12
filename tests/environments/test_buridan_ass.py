"""
Unit tests file where testing BuridanAss environment.
"""
import gym

import spaces
from environments import BuridanAss
from tests.environments.test_env_mesh import TestEnvMesh


class TestBuridanAss(TestEnvMesh):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = BuridanAss(seed=0)

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        super().test_init()

        # This environment must have another attributes
        self.assertTrue(hasattr(self.environment, 'p_stolen'))
        self.assertTrue(hasattr(self.environment, 'n_appear'))
        self.assertTrue(hasattr(self.environment, 'walking_penalty'))
        self.assertTrue(hasattr(self.environment, 'stolen_penalty'))
        self.assertTrue(hasattr(self.environment, 'hunger_penalty'))
        self.assertTrue(hasattr(self.environment, 'food_counter'))

        # By default initial position is (1, 1)
        self.assertEqual(((1, 1), {(0, 0), (2, 2)}, 0), self.environment.initial_state)

        # Observation space
        self.assertEqual(
            gym.spaces.Tuple(
                (
                    gym.spaces.Tuple((gym.spaces.Discrete(3), gym.spaces.Discrete(3))),
                    spaces.Bag([
                        frozenset(), frozenset({(0, 0)}), frozenset({(2, 2)}), frozenset({(0, 0), (2, 2)})
                    ]),
                    gym.spaces.Discrete(10)
                )
            ), self.environment.observation_space
        )

        # Default reward is (0., 0., 0.)
        self.assertEqual((0., 0., 0.), self.environment.default_reward)

        # Check if food counters are ok
        for position, food in self.environment.food_counter.items():
            self.assertTrue(self.environment.observation_space[0].contains(position))
            self.assertEqual(0, food)

    def test_reset(self):
        """
        Testing reset method
        :return:
        """

        # Set current position to random position
        self.environment.current_state = self.environment.observation_space.sample()

        # Set all food position to False
        for state in self.environment.food_counter.keys():
            self.environment.food_counter.update({state: 0})

        # Reset environment
        self.environment.reset()

        # Asserts
        self.assertEqual(self.environment.initial_state, self.environment.current_state)

        # Check if finals states are correct
        for position, food in self.environment.food_counter.items():
            self.assertTrue(self.environment.observation_space[0].contains(position))
            self.assertEqual(0, food)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        ################################################################################################################
        # Begin at position (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        state = ((0, 0), {(0, 0), (2, 2)}, 0)
        self.environment.current_state = state

        # Cannot go to UP (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(((0, 0), {(0, 0), (2, 2)}, 1), next_state)

        # Go to RIGHT (increment x axis)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(((1, 0), {(0, 0), (2, 2)}, 1), next_state)

        # Go to DOWN (increment y axis)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(((0, 1), {(0, 0), (2, 2)}, 1), next_state)

        # Cannot go to LEFT (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(((0, 0), {(0, 0), (2, 2)}, 1), next_state)

        ################################################################################################################
        # Set to (2, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = ((2, 0), {(0, 0), (2, 2)}, 0)
        self.environment.current_state = state

        # Cannot go to UP (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(((2, 0), {(0, 0), (2, 2)}, 1), next_state)

        # Cannot go to RIGHT (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(((2, 0), {(0, 0), (2, 2)}, 1), next_state)

        # Go to DOWN (increment y axis)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(((2, 1), {(0, 0), (2, 2)}, 1), next_state)

        # Go to LEFT (decrement x axis)
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(((1, 0), {(0, 0), (2, 2)}, 1), next_state)

        ################################################################################################################
        # Set to (2, 2) (DOWN-RIGHT corner)
        ################################################################################################################
        state = ((2, 2), {(0, 0), (2, 2)}, 0)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(((2, 1), {(0, 0), (2, 2)}, 1), next_state)

        # Cannot go to RIGHT (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(((2, 2), {(0, 0), (2, 2)}, 1), next_state)

        # Cannot go to DOWN (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(((2, 2), {(0, 0), (2, 2)}, 1), next_state)

        # Go to LEFT (decrement x axis)
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(((1, 2), {(0, 0), (2, 2)}, 1), next_state)

        ################################################################################################################
        # Set to (0, 2) (DOWN-LEFT corner)
        ################################################################################################################
        state = ((0, 2), {(0, 0), (2, 2)}, 0)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(((0, 1), {(0, 0), (2, 2)}, 1), next_state)

        # Go to RIGHT (increment x axis)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(((1, 2), {(0, 0), (2, 2)}, 1), next_state)

        # Cannot go to DOWN (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(((0, 2), {(0, 0), (2, 2)}, 1), next_state)

        # Cannot go to LEFT (Keep in same position
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(((0, 2), {(0, 0), (2, 2)}, 1), next_state)

        ################################################################################################################
        # Set to (1, 1) (CENTER)
        ################################################################################################################
        state = ((1, 1), {(0, 0), (2, 2)}, 0)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(((1, 0), {(0, 0), (2, 2)}, 1), next_state)

        # Go to RIGHT (increment x axis)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(((2, 1), {(0, 0), (2, 2)}, 1), next_state)

        # Go to DOWN (increment y axis)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(((1, 2), {(0, 0), (2, 2)}, 1), next_state)

        # Go to LEFT (decrement x axis)
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(((0, 1), {(0, 0), (2, 2)}, 1), next_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Disable probability to stole
        self.environment.p_stolen = 0

        # Simple valid step, at each step penalizes -1.
        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['DOWN'])

        # State:
        #   (position, states_visible_with_food, last_ate)
        # Reward:
        #   [hungry_penalize, stolen_penalize, step_penalize]

        # Remember that initial position is (1, 1), and this problem return a complex position
        self.assertEqual(((1, 2), {(0, 0), (2, 2)}, 1), next_state)
        self.assertEqual([0, 0, -1], reward)
        self.assertFalse(is_final)
        self.assertFalse(info)

        # Return at begin, to see both food stacks.
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual(((1, 1), {(0, 0), (2, 2)}, 2), next_state)
        self.assertEqual([0, 0, -1], reward)
        self.assertFalse(is_final)

        # Go to RIGHT
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual(((2, 1), {(0, 0), (2, 2)}, 3), next_state)
        self.assertEqual([0, 0., -1], reward)
        self.assertFalse(is_final)

        # Set probability to stolen on 1
        self.environment.p_stolen = 1.

        # Go to DOWN
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        # Food stack (0, 0) is stolen
        self.assertEqual(((2, 2), {(2, 2)}, 4), next_state)
        self.assertEqual([0, -0.5, -1], reward)
        self.assertFalse(is_final)

        # Go to STAY
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['STAY'])

        # Not more food stacks, donkey has ate.
        self.assertEqual(((2, 2), frozenset(), 0), next_state)
        self.assertEqual([0, 0, 0], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Disable probability to stole
        self.environment.p_stolen = 0

        # Wasteful steps for the donkey to be hungry
        for _ in range(3):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # Donkey has hungry.
        self.assertEqual(((2, 1), {(2, 2), (0, 0)}, 3), next_state)
        self.assertEqual([0.0, 0.0, -1.0], reward)
        self.assertFalse(is_final)

        for _ in range(7):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # Go to DOWN (2, 2)
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])
        self.assertEqual(((2, 2), {(2, 2), (0, 0)}, 9), next_state)
        self.assertEqual([-1, 0.0, -1.], reward)

        # Ensure probability to stolen
        self.environment.p_stolen = 1.

        # Go to STAY (2, 2)
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['STAY'])

        # Donkey has ate.
        self.assertEqual(((2, 2), frozenset(), 0), next_state)
        self.assertEqual([0, -0.5, 0], reward)
        self.assertTrue(is_final)

    def test_states(self):
        """
        Testing that all states must be contained into observation space
        :return:
        """
        self.assertTrue(
            all(
                self.environment.observation_space.contains(
                    state
                ) for state in self.environment.states()
            )
        )

    def test_states_size(self):
        self.assertEqual(len(self.environment.states()), 86)
