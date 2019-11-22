"""
Unit tests file where testing DeepSeaTreasure environment.
"""

from gym import spaces

from environments import DeepSeaTreasure
from models import Vector
from tests.environments.test_env_mesh import TestEnvMesh


class TestDeepSeaTreasure(TestEnvMesh):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = DeepSeaTreasure(seed=0)

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        super().test_init()

        # By default mesh shape is 10x11
        self.assertEqual(spaces.Tuple((spaces.Discrete(10), spaces.Discrete(11))), self.environment.observation_space)

        # By default initial position is (0, 0)
        self.assertEqual((0, 0), self.environment.initial_state)

        # Default reward is (-1, 0)
        self.assertEqual((-1, 0), self.environment.default_reward)

    def test_action_space_length(self):
        # By default action space is 4 (UP, RIGHT, DOWN, LEFT)
        self.assertEqual(self.environment.action_space.n, 4)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        ################################################################################################################
        # Begin at position (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        state = (0, 0)
        self.environment.current_state = state

        # Cannot go to UP (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(state, new_state)

        # Go to RIGHT (increment x axis)
        new_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((state[0] + 1, state[1]), new_state)

        # Go to DOWN (increment y axis)
        new_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((state[0], state[1] + 1), new_state)

        # Cannot go to LEFT (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(state, new_state)

        ################################################################################################################
        # Set to (9, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (9, 0)
        self.environment.current_state = state

        # Cannot go to UP (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(state, new_state)

        # Cannot go to RIGHT (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(state, new_state)

        # Go to DOWN (increment y axis)
        new_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((state[0], state[1] + 1), new_state)

        # Go to LEFT (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual((state[0] - 1, state[1]), new_state)

        ################################################################################################################
        # Set to (9, 10) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (9, 10)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        new_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual((state[0], state[1] - 1), new_state)

        # Cannot go to RIGHT (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(state, new_state)

        # Cannot go to DOWN (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(state, new_state)

        # Cannot go to LEFT (Keep in same position, obstacle)
        new_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(state, new_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Simple valid step
        # Reward:
        #   [time_inverted, treasure_value]
        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['DOWN'])

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
            _, _, _, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # Go to down 6 steps, until (6, 6).
        for _ in range(6):
            _, _, _, _ = self.environment.step(action=self.environment.actions['DOWN'])

        # Try to go LEFT, but is an obstacle.
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['LEFT'])

        self.assertEqual((6, 6), next_state)
        self.assertEqual([-1, 0], reward)
        self.assertFalse(is_final)

        # Go to DOWN to get 24 treasure-value.
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual((6, 7), next_state)
        self.assertEqual([-1, 24], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final position.
        # Go to right 9 steps, until (9, 0).
        for _ in range(9):
            _, _, _, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # Go to down 10 steps, until (9, 10).
        for _ in range(10):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual((9, 10), next_state)
        self.assertEqual([-1, 124], reward)
        self.assertTrue(is_final)

    def test_states_size(self):
        self.assertEqual(len(self.environment.states()), 51)

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
            ), Vector((-1, 2))
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

        # For any state the following happens
        for state in self.environment.states():

            # Decompose state
            x, y = state

            # If go to UP, or can go to UP or keep in same position
            reachable_states = self.environment.reachable_states(state=state, action=self.environment.actions['UP'])

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x, y - 1), (x, y)})

            # Go to right
            reachable_states = self.environment.reachable_states(state=state, action=self.environment.actions['RIGHT'])

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x + 1, y), (x, y)})

            # Go to down
            reachable_states = self.environment.reachable_states(state=state, action=self.environment.actions['DOWN'])

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x, y + 1), (x, y)})

            # Go to left
            reachable_states = self.environment.reachable_states(state=state, action=self.environment.actions['LEFT'])

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x - 1, y), (x, y)})
