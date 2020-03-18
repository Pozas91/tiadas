"""
Unit tests path where testing MoPuddleWorld environment.
"""

from environments import MoPuddleWorldAcyclic
from tests.environments.test_mo_puddleworld import TestMoPuddleWorld


class TestMoPuddleWorldAcyclic(TestMoPuddleWorld):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = MoPuddleWorldAcyclic(seed=0)

    def test_action_space_length(self):
        # By default action space is 2 (UP, RIGHT)
        self.assertEqual(len(self.environment.actions), 2)

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

        self.assertTrue(self.environment.action_space.contains(
            self.environment.actions['RIGHT']
        ))

        # Go to RIGHT (increment x axis)
        new_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((state[0] + 1, state[1]), new_state)

        ################################################################################################################
        # Set to (19, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (19, 0)
        self.environment.current_state = state

        self.assertTrue(self.environment.action_space.n == 0)

        ################################################################################################################
        # Set to (19, 19) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (19, 19)
        self.environment.current_state = state

        self.assertTrue(
            self.environment.action_space.n == 1 and
            self.environment.action_space.contains(self.environment.actions['UP'])
        )

        # Go to UP (decrement y axis)
        new_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual((state[0], state[1] - 1), new_state)

        ################################################################################################################
        # Set to (0, 19) (DOWN-LEFT corner)
        ################################################################################################################
        state = (0, 19)
        self.environment.current_state = state

        self.assertTrue(
            self.environment.action_space.n == 2 and
            self.environment.action_space.contains(self.environment.actions['UP']) and
            self.environment.action_space.contains(self.environment.actions['RIGHT'])
        )

        # Go to UP (decrement y axis)
        new_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual((state[0], state[1] - 1), new_state)

        # Go to RIGHT (increment x axis)
        new_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((state[0] + 1, state[1]), new_state)

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
        self.environment.current_state = (0, 13)

        # Enter to a puddle
        for _ in range(8):
            next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['RIGHT'])

        # Remember that initial position is (0, 13)
        self.assertEqual((8, 13), next_state)
        self.assertEqual([-1, -2], reward)
        self.assertFalse(is_final)
        self.assertFalse(info)

        # Enter in puddles a little more.
        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual((8, 12), next_state)
        self.assertEqual([-1, -2], reward)

        # Enter in puddles a little more.
        next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual((8, 11), next_state)
        self.assertEqual([-1, -2], reward)

        # Enter a more...
        for _ in range(7):
            next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual((8, 4), next_state)
        self.assertEqual([-1, -3], reward)

        ################################################################################################################

        # Go to final position
        for _ in range(11):
            _ = self.environment.step(action=self.environment.actions['RIGHT'])

        for _ in range(4):
            next_state, reward, is_final, info = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual((19, 0), next_state)
        self.assertEqual([10, 0], reward)
        self.assertTrue(is_final)

    def test_states_size(self):
        self.assertEqual(len(self.environment.states()), 399)

    def test_reachable_states(self):

        # For any state the following happens
        for state in self.environment.states():
            # Decompose state
            x, y = state

            if x >= (self.environment.observation_space[0].n - 1):
                with self.assertRaises(ValueError):
                    self.environment.reachable_states(action=self.environment.actions['RIGHT'], state=state)
            else:
                # Go to right
                reachable_states = self.environment.reachable_states(
                    state=state, action=self.environment.actions['RIGHT']
                )

                self.assertTrue(len(reachable_states) == 1)
                self.assertIn(next(iter(reachable_states)), {(x + 1, y), (x, y)})

            if y <= 0:
                with self.assertRaises(ValueError):
                    self.environment.reachable_states(action=self.environment.actions['UP'], state=state)
            else:
                # Go to up
                reachable_states = self.environment.reachable_states(
                    state=state, action=self.environment.actions['UP']
                )

                self.assertTrue(len(reachable_states) == 1)
                self.assertIn(next(iter(reachable_states)), {(x, y - 1), (x, y)})
