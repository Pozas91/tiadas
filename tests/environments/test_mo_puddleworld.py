"""
Unit tests path where testing MoPuddleWorld environment.
"""

from gym import spaces

from environments import MoPuddleWorld
from models import VectorDecimal
from tests.environments.test_env_mesh import TestEnvMesh


class TestMoPuddleWorld(TestEnvMesh):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = MoPuddleWorld(seed=0)

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # By default mesh shape is 20x20
        self.assertEqual(spaces.Tuple((spaces.Discrete(20), spaces.Discrete(20))), self.environment.observation_space)

        # Default reward is (10, 0)
        self.assertEqual((10, 0), self.environment.default_reward)

    def test_action_space_length(self):
        # By default action space is 4 (UP, RIGHT, DOWN, LEFT)
        self.assertEqual(self.environment.action_space.n, 4)

    def test_reset(self):
        """
        Testing reset method
        :return:
        """

        # Reset environment
        self.environment.reset()

        # Asserts
        self.assertTrue(self.environment.observation_space.contains(self.environment.current_state))
        self.assertFalse(self.environment.current_state == self.environment.final_state)

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
        # Set to (19, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (19, 0)
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
        # Set to (19, 19) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (19, 19)
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

        # Cannot go to LEFT (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual((state[0] - 1, state[1]), new_state)

        ################################################################################################################
        # Set to (0, 19) (DOWN-LEFT corner)
        ################################################################################################################
        state = (0, 19)
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        new_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual((state[0], state[1] - 1), new_state)

        # Go to RIGHT (increment x axis)
        new_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((state[0] + 1, state[1]), new_state)

        # Cannot go to DOWN (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(state, new_state)

        # Cannot go to LEFT (Keep in same position)
        new_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(state, new_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # THIS ENVIRONMENT HAS A RANDOM INITIAL STATE. FOR TESTING I USE A PREDEFINED INITIAL STATE.
        # Reward:
        #   [non_goal_reached, puddle_penalize]

        next_state = None
        reward = None
        is_final = False
        info = {}

        # Set a current position
        self.environment.current_state = (0, 0)

        # Simple valid step
        for _ in range(3):
            next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['DOWN'])

        # Remember that initial position is (0, 0)
        self.assertEqual((0, 3), next_state)
        self.assertEqual([-1, -1], reward)
        self.assertFalse(is_final)
        self.assertFalse(info)

        # Enter in puddles a little more.
        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual((0, 4), next_state)
        self.assertEqual([-1, -2], reward)

        # Enter in puddles a little more.
        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual((0, 5), next_state)
        self.assertEqual([-1, -2], reward)

        # Enter a more...
        for _ in range(7):
            next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual((7, 5), next_state)
        self.assertEqual([-1, -4], reward)

        ################################################################################################################

        # Go to final position
        for _ in range(12):
            _ = self.environment.step(action=self.environment.actions['RIGHT'])

        for _ in range(5):
            next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual((19, 0), next_state)
        self.assertEqual([10, 0], reward)
        self.assertTrue(is_final)

    def test_states_size(self):
        self.assertEqual(len(self.environment.states()), 399)

    def test_transition_reward(self):

        # In this environment doesn't mind initial state to get the reward
        state = self.environment.observation_space.sample()

        # Doesn't mind action too.
        action = self.environment.action_space.sample()

        # A non-puddle state
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(1, 0)
            ), VectorDecimal((-1, 0))
        )

        # Another non-puddle state
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(4, 10)
            ), VectorDecimal((-1, 0))
        )

        # Another non-puddle state
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(15, 12)
            ), VectorDecimal((-1, 0))
        )

        # State in a border of puddle
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(9, 2)
            ), VectorDecimal((-1, -1))
        )

        # Another state in a border of puddle
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(9, 10)
            ), VectorDecimal((-1, -1))
        )

        # State in a puddle
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(8, 10)
            ), VectorDecimal((-1, -2))
        )

        # State in a corner of puddle
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(9, 3)
            ), VectorDecimal((-1, -2))
        )

        # Final state
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(19, 0)
            ), VectorDecimal((10, 0))
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
