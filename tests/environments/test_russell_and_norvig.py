"""
Unit tests file where testing RusselAndNorvig environment.
"""

from gym import spaces

import utils.environments as ue
from environments import RussellNorvig
from models import VectorDecimal
from tests.environments.test_env_mesh import TestEnvMesh


class TestRusselAndNorvig(TestEnvMesh):

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = RussellNorvig(seed=0)

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # This environment must have another attributes
        self.assertTrue(hasattr(self.environment, 'transitions'))

        # By default mesh shape is 4x3
        self.assertEqual(spaces.Tuple((spaces.Discrete(4), spaces.Discrete(3))), self.environment.observation_space)

        # By default initial position is (0, 2)
        self.assertEqual((0, 2), self.environment.initial_state)

        # Default reward is (-0.04)
        self.assertEqual((-0.04,), self.environment.default_reward)

    def test_action_space_length(self):
        self.assertEqual(4, len(self.environment.actions))

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
        # Set to (3, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (3, 0)
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
        # Set to (3, 2) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (3, 2)
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

        # Go to LEFT (decrement x axis)
        new_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual((state[0] - 1, state[1]), new_state)

        ################################################################################################################
        # Set to (0, 2) (DOWN-LEFT corner)
        ################################################################################################################
        state = (0, 2)
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

        # Reward:
        #   [time_inverted, treasure_value]
        # Remember that initial position is (0, 2)

        # To testing force to go always in desired direction.
        self.environment.transitions = [1, 0, 0, 0]

        # Reset environment
        self.environment.reset()

        # Go to RIGHT 3 steps, until (3, 2).
        for _ in range(3):
            _ = self.environment.step(action=self.environment.actions['RIGHT'])

        new_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual((3, 1), new_state)
        self.assertEqual([-1], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to another final position.
        # Go to UP 9 steps, until (0, 0).
        for _ in range(2):
            _ = self.environment.step(action=self.environment.actions['UP'])

        # Go to RIGHT 3 steps, until (3, 0).
        for _ in range(3):
            new_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual((3, 0), new_state)
        self.assertEqual([1], reward)
        self.assertTrue(is_final)

        ################################################################################################################
        # Testing p_stochastic
        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Force DIR_90 transition
        self.environment.transitions = [0, 1, 0, 0]

        # Agent turns to RIGHT
        new_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual((1, 2), new_state)

    def test_states_size(self):
        self.assertEqual(len(self.environment.states()), 9)

    def test_transition_probability(self):

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

                    if ue.is_on_up(state=state, next_state=next_state):
                        self.assertEqual(self.environment.transitions[(coefficient + 0) % n_actions], probability)
                    elif ue.is_on_right(state=state, next_state=next_state):
                        self.assertEqual(self.environment.transitions[(coefficient + 1) % n_actions], probability)
                    elif ue.is_on_down(state=state, next_state=next_state):
                        self.assertEqual(self.environment.transitions[(coefficient + 2) % n_actions], probability)
                    elif ue.is_on_left(state=state, next_state=next_state):
                        self.assertEqual(self.environment.transitions[(coefficient + 3) % n_actions], probability)

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

    def test_transition_reward(self):

        # In this environment doesn't mind initial state to get the reward
        state = self.environment.observation_space.sample()

        # Doesn't mind action too.
        action = self.environment.action_space.sample()

        # An intermediate state
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(1, 0)
            ), self.environment.default_reward
        )

        # A final state
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(3, 0)
            ), VectorDecimal((1,))
        )

        # Another final state
        self.assertEqual(
            self.environment.transition_reward(
                state=state, action=action, next_state=(3, 1)
            ), VectorDecimal((-1,))
        )
