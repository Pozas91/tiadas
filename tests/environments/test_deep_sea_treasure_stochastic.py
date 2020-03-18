"""
Unit tests path where testing DeepSeaTreasure environment.
"""

import utils.environments as ue
from environments import DeepSeaTreasureStochastic
from tests.environments.test_deep_sea_treasure import TestDeepSeaTreasure


class TestDeepSeaTreasureStochastic(TestDeepSeaTreasure):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = DeepSeaTreasureStochastic(seed=0)

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        super().__init__()

        # This environment must have another attributes
        self.assertTrue(hasattr(self.environment, 'transitions'))

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Disable p_stochastic
        self.environment.transitions = [1, 0, 0, 0]

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

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Enable p_stochastic (In this case always turn to direction + DIR_90, 100% probability)
        self.environment.transitions = [0, 1, 0, 0]

        # Simple valid step
        # Reward:
        #   [time_inverted, treasure_value]

        # I want to go DOWN, but DOWN + DIR_90 = LEFT
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        # Remember that initial position is (0, 0)
        self.assertEqual((0, 0), next_state)
        self.assertEqual([-1, 0], reward)
        self.assertFalse(is_final)

        # If we want to go to RIGHT, then we must choose UP direction
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual((0, 1), next_state)
        self.assertEqual([-1, 1], reward)
        self.assertTrue(is_final)

    def test_reachable_states(self):

        # For any state the following happens
        for state in self.environment.states():
            # Decompose state
            x, y = state

            self.environment.current_state = state
            for action in self.environment.action_space:
                # If go to UP, or can go to UP or keep in same position
                reachable_states = self.environment.reachable_states(state=state, action=action)

                # At least 3 states are available
                self.assertTrue(3 <= len(reachable_states) <= 4)

                self.assertTrue(
                    # Or go to UP
                    (x, y - 1) in reachable_states or
                    # Or go to left
                    (x - 1, y) in reachable_states or
                    # Or got to right
                    (x + 1, y) in reachable_states or
                    # Or got to down
                    (x, y + 1) in reachable_states or
                    # Or keep in same position
                    (x, y) in reachable_states
                )

    def test_transition_probability(self):

        # Get actions
        action_up = self.environment.actions['UP']
        action_down = self.environment.actions['DOWN']
        action_right = self.environment.actions['RIGHT']
        action_left = self.environment.actions['LEFT']

        # For all states, for all actions and for all next_state possibles, transition probability must be return 1.
        for state in self.environment.states():

            # Set state as current state
            self.environment.current_state = state

            for action in self.environment.action_space:

                for next_state in self.environment.reachable_states(state=state, action=action):

                    probability = self.environment.transition_probability(
                        state=state, action=action, next_state=next_state
                    )

                    n_actions = len(self.environment.actions)
                    coefficient = (n_actions - action)

                    if action == action_up and ue.is_on_up_or_same_position(state=state, next_state=next_state):
                        self.assertEqual(self.environment.transitions[(coefficient + 0) % n_actions], probability)
                    elif action == action_right and ue.is_on_right_or_same_position(state=state, next_state=next_state):
                        self.assertEqual(self.environment.transitions[(coefficient + 1) % n_actions], probability)
                    elif action == action_down and ue.is_on_down_or_same_position(state=state, next_state=next_state):
                        self.assertEqual(self.environment.transitions[(coefficient + 2) % n_actions], probability)
                    elif action == action_left and ue.is_on_left_or_same_position(state=state, next_state=next_state):
                        self.assertEqual(self.environment.transitions[(coefficient + 3) % n_actions], probability)
