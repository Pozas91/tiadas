"""
Unit tests path where testing NonRecurrentRings environment.
"""

from gym import spaces

from environments import NonRecurrentRings
from tests.environments.test_environment import TestEnvironment


class TestNonRecurrentRings(TestEnvironment):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = NonRecurrentRings(seed=0)

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        super().__init__()

        # Observation space is 7 states
        self.assertEqual(spaces.Discrete(8), self.environment.observation_space)

        self.assertTrue(len(self.environment.actions) == 2)

        # By default initial position is 0
        self.assertEqual(0, self.environment.initial_state)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        # Set position
        self.environment.current_state = 0

        # Go to COUNTER_CLOCKWISE sense
        new_state = self.environment.next_state(action=self.environment.actions['CLOCKWISE'])

        # State 8
        self.assertEqual(7, new_state)

        ################################################################################################################

        # Set position
        self.environment.current_state = 7

        # Go to COUNTER_CLOCKWISE sense
        new_state = self.environment.next_state(action=self.environment.actions['COUNTER_CLOCKWISE'])

        # State 1
        self.assertEqual(0, new_state)

        ################################################################################################################

        # Set position
        self.environment.current_state = 0

        # Go to COUNTER_CLOCKWISE sense
        new_state = self.environment.next_state(action=self.environment.actions['COUNTER_CLOCKWISE'])

        # State 2
        self.assertEqual(1, new_state)

        ################################################################################################################

        # Set position
        self.environment.current_state = 1

        # Go to COUNTER_CLOCKWISE sense
        new_state = self.environment.next_state(action=self.environment.actions['COUNTER_CLOCKWISE'])

        # State 3
        self.assertEqual(2, new_state)

        ################################################################################################################

        # Set position
        self.environment.current_state = 3

        # Go to COUNTER_CLOCKWISE sense from position 3.
        new_state = self.environment.next_state(action=self.environment.actions['COUNTER_CLOCKWISE'])

        # State 1
        self.assertEqual(0, new_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # THIS ENVIRONMENT DOES NOT HAVE FINAL STEP (is_final is always False)
        # Reward:
        #   [value_1, value_2]

        # Simple valid step, begin at position 0
        next_state, reward, is_final, info = self.environment.step(
            action=self.environment.actions['COUNTER_CLOCKWISE']
        )

        self.assertEqual(1, next_state)
        self.assertEqual((2, -1), reward)
        self.assertFalse(is_final)
        self.assertFalse(info)

        ################################################################################################################

        # State 2
        next_state, reward, is_final, info = self.environment.step(
            action=self.environment.actions['COUNTER_CLOCKWISE']
        )

        self.assertEqual(2, next_state)
        self.assertEqual((2, -1), reward)

        # State 3
        next_state, reward, is_final, info = self.environment.step(
            action=self.environment.actions['COUNTER_CLOCKWISE']
        )

        self.assertEqual(3, next_state)
        self.assertEqual((2, -1), reward)

        # State 4
        next_state, reward, is_final, info = self.environment.step(
            action=self.environment.actions['COUNTER_CLOCKWISE']
        )

        self.assertEqual(0, next_state)
        self.assertEqual((2, -1), reward)

        # State 1
        _ = self.environment.step(
            action=self.environment.actions['COUNTER_CLOCKWISE']
        )

        next_state, reward, is_final, info = self.environment.step(
            action=self.environment.actions['CLOCKWISE']
        )

        self.assertEqual(0, next_state)
        self.assertEqual((-1, 0), reward)

        # State 1
        next_state, reward, is_final, info = self.environment.step(
            action=self.environment.actions['CLOCKWISE']
        )

        self.assertEqual(7, next_state)
        self.assertEqual((-1, 0), reward)

        # State 8
        next_state, reward, is_final, info = self.environment.step(
            action=self.environment.actions['CLOCKWISE']
        )

        self.assertEqual(4, next_state)
        self.assertEqual((-1, 2), reward)

        # State 5
        next_state, reward, is_final, info = self.environment.step(
            action=self.environment.actions['CLOCKWISE']
        )

        self.assertEqual(5, next_state)
        self.assertEqual((-1, 2), reward)

        # State 6
        next_state, reward, is_final, info = self.environment.step(
            action=self.environment.actions['CLOCKWISE']
        )

        self.assertEqual(6, next_state)
        self.assertEqual((-1, 2), reward)

        # State 7
        next_state, reward, is_final, info = self.environment.step(
            action=self.environment.actions['CLOCKWISE']
        )

        self.assertEqual(7, next_state)
        self.assertEqual((-1, 2), reward)

        # State 8
        _ = self.environment.step(
            action=self.environment.actions['CLOCKWISE']
        )

        next_state, reward, is_final, info = self.environment.step(
            action=self.environment.actions['COUNTER_CLOCKWISE']
        )

        self.assertEqual(7, next_state)
        self.assertEqual((0, -1), reward)

    def test_reachable_states(self):
        # For any state the following happens
        for state in self.environment.states():

            # Set state as current state
            self.environment.current_state = state

            for action in self.environment.action_space:
                reachable_states = self.environment.reachable_states(state=state, action=action)

                # Must be only one reachable state
                self.assertTrue(len(reachable_states) == 1)

                self.assertEqual(next(iter(reachable_states)), self.environment.possible_transitions[state][action])

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

    def test_transition_reward(self):
        # For any state the following happens
        for state in self.environment.states():

            # Set state as current state
            self.environment.current_state = state

            for action in self.environment.action_space:
                reachable_states = self.environment.reachable_states(state=state, action=action)

                for next_state in reachable_states:

                    # Get transition reward
                    reward = self.environment.transition_reward(state=state, action=action, next_state=next_state)

                    self.assertEqual(reward, self.environment.rewards_dictionary[state][action])
