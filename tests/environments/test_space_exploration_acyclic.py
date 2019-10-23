"""
Unit tests file where testing SpaceExploration environment.
"""

import utils.environments as ue
from environments import SpaceExplorationAcyclic
from tests.environments.test_space_exploration import TestSpaceExploration


class TestSpaceExplorationAcyclic(TestSpaceExploration):

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = SpaceExplorationAcyclic(seed=0)

    def test_action_space_length(self):
        self.assertEqual(len(self.environment.actions), 3)

    def test_initial_state(self):
        # By default initial position is (0, 0)
        self.assertEqual((0, 0), self.environment.initial_state)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        # THIS IS A ACYCLIC ENVIRONMENT

        ################################################################################################################
        # Begin at position (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        state = (0, 0)
        self.environment.current_state = state

        # Go to RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((1, 0), next_state)

        # Go to DOWN RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN RIGHT'])
        self.assertEqual((1, 1), next_state)

        # Go to DOWN
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((0, 1), next_state)

        ################################################################################################################
        # Begin at position (12, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (12, 0)
        self.environment.current_state = state

        # Go to RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((0, 0), next_state)

        # Go to DOWN RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN RIGHT'])
        self.assertEqual((0, 1), next_state)

        # Go to DOWN
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((12, 1), next_state)

        ################################################################################################################
        # Begin at position (12, 4) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (12, 4)
        self.environment.current_state = state

        # Go to RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((0, 4), next_state)

        # Go to DOWN RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN RIGHT'])
        self.assertEqual((0, 0), next_state)

        # Go to DOWN
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((12, 0), next_state)

        ################################################################################################################
        # Begin at position (0, 4) (DOWN-LEFT corner)
        ################################################################################################################
        state = (0, 4)
        self.environment.current_state = state

        # Go to RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((1, 4), next_state)

        # Go to DOWN RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN RIGHT'])
        self.assertEqual((1, 0), next_state)

        # Go to DOWN
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((0, 0), next_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Reward:
        #   [mission_success, radiation]
        # Remember that initial position is (0, 0)

        # Reset environment
        self.environment.reset()

        next_state, reward, is_final = None, None, None

        # Go to RIGHT 5 steps, until (5, 0), ship is destroyed by asteroid.
        for _ in range(5):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual((5, 0), next_state)
        self.assertEqual([-100, -1], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to a final position.
        # (1, 1)
        _ = self.environment.step(action=self.environment.actions['DOWN RIGHT'])

        # (5, 1)
        for _ in range(4):
            _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # (6, 2)
        _ = self.environment.step(action=self.environment.actions['DOWN RIGHT'])

        # (7, 3), position with radiation
        _ = self.environment.step(action=self.environment.actions['DOWN RIGHT'])

        # (10, 3)
        for _ in range(3):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual((10, 3), next_state)
        self.assertEqual([0, -11], reward)
        self.assertFalse(is_final)

        # (12, 3)
        for _ in range(2):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual((12, 3), next_state)
        self.assertEqual([30, -1], reward)
        self.assertTrue(is_final)

    def test_states_size(self):
        self.assertEqual(len(self.environment.states()), 57)

    def test_reachable_states(self):

        # For any state the following happens
        for state in self.environment.states():
            # Decompose state
            x, y = state

            ############################################################################################################
            # Go to right
            reachable_states = self.environment.reachable_states(state=state, action=self.environment.actions['RIGHT'])

            x_moved = ue.move_right(x=x, limit=self.environment.observation_space[0].n)
            y_moved = y

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x_moved, y_moved)})
            ############################################################################################################
            # Go to down-right
            reachable_states = self.environment.reachable_states(
                state=state, action=self.environment.actions['DOWN RIGHT']
            )

            x_moved = ue.move_right(x=x, limit=self.environment.observation_space[0].n)
            y_moved = ue.move_down(y=y, limit=self.environment.observation_space[1].n)

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x_moved, y_moved)})
            ############################################################################################################
            # Go to down
            reachable_states = self.environment.reachable_states(
                state=state, action=self.environment.actions['DOWN']
            )

            x_moved = x
            y_moved = ue.move_down(y=y, limit=self.environment.observation_space[1].n)

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x_moved, y_moved)})
