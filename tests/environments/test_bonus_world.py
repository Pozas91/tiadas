"""
Unit tests file where testing BonusWorld environment.
"""

import gym

import spaces
from environments import BonusWorld
from models import Vector
from tests.environments.test_env_mesh import TestEnvMesh


class TestBonusWorld(TestEnvMesh):
    environment = None

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = BonusWorld(seed=0)

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # This environment must have another attributes
        self.assertTrue(hasattr(self.environment, 'pits'))
        self.assertTrue(hasattr(self.environment, 'bonus'))

        self.assertEqual(
            gym.spaces.Tuple(
                (
                    gym.spaces.Tuple((gym.spaces.Discrete(9), gym.spaces.Discrete(9))),
                    spaces.Boolean()
                )
            ), self.environment.observation_space
        )

        # By default initial position is (0, 0)
        self.assertEqual(((0, 0), False), self.environment.initial_state)

        # Default reward is (0, 0, -1)
        self.assertEqual((0, 0, -1), self.environment.default_reward)

    def test_action_space_length(self):
        # By default action space is 4 (UP, RIGHT, DOWN, LEFT)
        self.assertEqual(len(self.environment.actions), 4)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        ################################################################################################################
        # Begin at position (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        position = (0, 0)

        # Cannot go to UP (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(position, next_state[0])

        # Go to RIGHT (increment x axis)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((position[0] + 1, position[1]), next_state[0])

        # Go to DOWN (increment y axis)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((position[0], position[1] + 1), next_state[0])

        # Cannot go to LEFT (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(position, next_state[0])

        ################################################################################################################
        # Set to (8, 0) (TOP-RIGHT corner)
        ################################################################################################################
        position = (8, 0)
        self.environment.current_state = position, False

        # Cannot go to UP (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(position, next_state[0])

        # Cannot go to RIGHT (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(position, next_state[0])

        # Go to DOWN (increment y axis)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((position[0], position[1] + 1), next_state[0])

        # Go to LEFT (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual((position[0] - 1, position[1]), next_state[0])

        ################################################################################################################
        # Set to (8, 8) (DOWN-RIGHT corner)
        ################################################################################################################
        position = (8, 8)
        self.environment.current_state = position, False

        # Go to UP (decrement y axis)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual((position[0], position[1] - 1), next_state[0])

        # Cannot go to RIGHT (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(position, next_state[0])

        # Cannot go to DOWN (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(position, next_state[0])

        # Go to LEFT (decrement x axis)
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual((position[0] - 1, position[1]), next_state[0])

        ################################################################################################################
        # Set to (0, 8) (DOWN-LEFT corner)
        ################################################################################################################
        position = (0, 8)
        self.environment.current_state = position, False

        # Go to UP (decrement y axis)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual((position[0], position[1] - 1), next_state[0])

        # Go to RIGHT (increment x axis)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((position[0] + 1, position[1]), next_state[0])

        # Cannot go to DOWN (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(position, next_state[0])

        # Cannot go to LEFT (Keep in same position
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(position, next_state[0])

        ################################################################################################################
        # Obstacles (For example, (2, 2)
        ################################################################################################################

        # Set to (2, 1)
        position = (2, 1)
        self.environment.current_state = position, False

        # Cannot go to DOWN (Keep in same position), because there is an obstacle.
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(position, next_state[0])

        # Set to (1, 2)
        position = (1, 2)
        self.environment.current_state = position, False

        # Cannot go to RIGHT (Keep in same position), because there is an obstacle.
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(position, next_state[0])

        # Set to (2, 1)
        position = (2, 1)
        self.environment.current_state = position, False

        # Cannot go to DOWN (Keep in same position), because there is an obstacle.
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(position, next_state[0])

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Simple valid step, at each step penalizes -1.
        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['DOWN'])

        # Remember that initial position is (0, 0)
        self.assertEqual(((0, 1), False), next_state)
        self.assertEqual([0, 0, -1], reward)
        self.assertFalse(is_final)
        self.assertFalse(info)

        # Do 7 steps more to reach final step (0, 8), which reward is (9, 1)
        for _ in range(7):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual(((0, 8), False), next_state)
        self.assertEqual([9, 1, -1], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final step
        for _ in range(8):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual(((8, 0), False), next_state)
        self.assertEqual([1, 9, -1], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to a PIT position (and reset to initial_state)
        _, _, _, _ = self.environment.step(action=self.environment.actions['DOWN'])

        for _ in range(7):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual(((0, 0), False), next_state)
        self.assertEqual([0, 0, -1], reward)
        self.assertFalse(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # At first bonus is disabled
        self.assertFalse(self.environment.current_state[1])

        # Get bonus and go to final position

        # 4 steps to RIGHT
        for _ in range(4):
            _, _, _, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # 3 steps to DOWN
        for _ in range(3):
            _, _, _, _ = self.environment.step(action=self.environment.actions['DOWN'])

        # 1 step to LEFT
        for _ in range(1):
            _, _, _, _ = self.environment.step(action=self.environment.actions['LEFT'])

        # Now bonus is activated
        self.assertTrue(self.environment.observation_space[1])

        # Go to final position with bonus activated

        # 3 steps to DOWN
        for _ in range(3):
            _, _, _, _ = self.environment.step(action=self.environment.actions['DOWN'])

        # 1 step to RIGHT
        for _ in range(1):
            _, _, _, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # 2 steps to DOWN
        for _ in range(2):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        # Final position (4, 8), which reward is (9, 5), but bonus is activated.
        self.assertEqual(((4, 8), True), next_state)
        self.assertEqual([9 * 2, 5 * 2, -1], reward)
        self.assertTrue(is_final)

    def test_states_size(self):
        self.assertEqual(len(self.environment.states()), 125)

    def test_transition_reward(self):

        # In this environment doesn't mind initial state to get the reward
        state = self.environment.observation_space.sample()

        # Doesn't mind action too.
        action = self.environment.action_space.sample()

        # An any state
        self.assertEqual(
            self.environment.default_reward,
            self.environment.transition_reward(
                state=state, action=action, next_state=((1, 2), False)
            )
        )

        # A final state
        self.assertEqual(
            Vector([1, 9, -1]),
            self.environment.transition_reward(
                state=state, action=action, next_state=((8, 0), False)
            )
        )

        # Same final state with bonus activated
        self.assertEqual(
            Vector([2, 18, -1]),
            self.environment.transition_reward(
                state=state, action=action, next_state=((8, 0), True)
            )
        )

        # Another final state
        self.assertEqual(
            Vector([9, 9, -1]),
            self.environment.transition_reward(
                state=state, action=action, next_state=((8, 8), False)
            )
        )

        # Same final state with bonus activated
        self.assertEqual(
            Vector([18, 18, -1]),
            self.environment.transition_reward(
                state=state, action=action, next_state=((8, 8), True)
            )
        )

    def test_transition_probability(self):

        # For all states, for all actions and for all next_state possibles, transition probability must be return 1.
        for state in self.environment.states():

            # Set state as current state
            self.environment.current_state = state

            for action in self.environment.action_space:
                for next_state in self.environment.reachable_states(state=state, action=action):

                    self.assertEqual(
                        1.,
                        self.environment.transition_probability(state=state, action=action, next_state=next_state)
                    )

    def test_reachable_states(self):

        reachable_states = self.environment.reachable_states(
            self.environment.initial_state, action=self.environment.actions['UP']
        )

        # Only one state (initial state), keep it in same state.
        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(self.environment.initial_state, reachable_states)

        reachable_states = self.environment.reachable_states(
            self.environment.initial_state, action=self.environment.actions['LEFT']
        )

        # Only one state (initial state), keep it in same state.
        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(self.environment.initial_state, reachable_states)

        reachable_states = self.environment.reachable_states(
            self.environment.initial_state, action=self.environment.actions['RIGHT']
        )

        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(((1, 0), False), reachable_states)

        reachable_states = self.environment.reachable_states(
            self.environment.initial_state, action=self.environment.actions['DOWN']
        )

        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(((0, 1), False), reachable_states)

        # Go to bonus
        reachable_states = self.environment.reachable_states(
            ((3, 4), False), action=self.environment.actions['UP']
        )

        # Bonus has been activated
        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(((3, 3), True), reachable_states)

        # Go to PIT
        reachable_states = self.environment.reachable_states(
            ((0, 7), False), action=self.environment.actions['RIGHT']
        )

        # Reset to initial state
        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(self.environment.initial_state, reachable_states)

        # Go to PIT, with bonus enabled
        reachable_states = self.environment.reachable_states(
            ((0, 7), True), action=self.environment.actions['RIGHT']
        )

        # Reset to initial state
        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(self.environment.initial_state, reachable_states)

        # Go to a final state
        reachable_states = self.environment.reachable_states(
            ((0, 7), True), action=self.environment.actions['DOWN']
        )

        # Reset to initial state
        self.assertTrue(len(reachable_states) == 1)
        self.assertIn(((0, 8), True), reachable_states)
