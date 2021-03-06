"""
Unit tests path where testing test ResourceGathering environment.
"""
import gym

from environments import ResourceGathering
from tests.environments.test_env_mesh import TestEnvMesh


class TestResourceGathering(TestEnvMesh):

    def setUp(self):
        # Set initial_seed to 0 to testing.
        self.environment = ResourceGathering(seed=0)

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        # This environment must have another attributes
        self.assertTrue(hasattr(self.environment, 'gold_positions'))
        self.assertTrue(hasattr(self.environment, 'gem_positions'))
        self.assertTrue(hasattr(self.environment, 'enemies_positions'))
        self.assertTrue(hasattr(self.environment, 'home_position'))

        # Observation space
        self.assertEqual(
            gym.spaces.Tuple(
                (
                    gym.spaces.Tuple((gym.spaces.Discrete(5), gym.spaces.Discrete(5))),
                    gym.spaces.Tuple((gym.spaces.Discrete(2), gym.spaces.Discrete(2))),
                )
            ), self.environment.observation_space
        )

        # By default initial position is (2, 4)
        self.assertEqual(((2, 4), (0, 0)), self.environment.initial_state)

        # Default reward is (0, 0, 0)
        self.assertEqual((0, 0, 0), self.environment.default_reward)

    def test_reset(self):
        """
        Testing reset method
        :return:
        """

        # Set current position to random position
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
        # Begin at position (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        self.environment.reset()

        state = ((0, 0), (0, 0))
        self.environment.current_state = state

        # Cannot go to UP (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(state, next_state)

        # Go to RIGHT (increment x axis)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(((1, 0), (0, 0)), next_state)

        # Go to DOWN (increment y axis)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(((0, 1), (0, 0)), next_state)

        # Cannot go to LEFT (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(state, next_state)

        ################################################################################################################
        # Set to (4, 0) (TOP-RIGHT corner)
        ################################################################################################################
        self.environment.reset()

        state = ((4, 0), (0, 0))
        self.environment.current_state = state

        # Cannot go to UP (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(state, next_state)

        # Cannot go to RIGHT (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(state, next_state)

        # Go to DOWN (increment y axis)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(((4, 1), (0, 1)), next_state)

        # Go to LEFT (decrement x axis) (enemy)
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(((2, 4), (0, 0)), next_state)

        ################################################################################################################
        # Set to (4, 4) (DOWN-RIGHT corner)
        ################################################################################################################
        self.environment.reset()

        state = ((4, 4), (0, 0))
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(((4, 3), (0, 0)), next_state)

        # Cannot go to RIGHT (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(state, next_state)

        # Cannot go to DOWN (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(state, next_state)

        # Go to LEFT (decrement x axis)
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(((3, 4), (0, 0)), next_state)

        ################################################################################################################
        # Set to (0, 4) (DOWN-LEFT corner)
        ################################################################################################################
        self.environment.reset()

        state = ((0, 4), (0, 0))
        self.environment.current_state = state

        # Go to UP (decrement y axis)
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(((0, 3), (0, 0)), next_state)

        # Go to RIGHT (increment x axis)
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(((1, 4), (0, 0)), next_state)

        # Cannot go to DOWN (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual(state, next_state)

        # Cannot go to LEFT (Keep in same position)
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual(state, next_state)

        ################################################################################################################
        # Set to (1, 0) and go to get gold
        ################################################################################################################
        self.environment.reset()

        state = ((1, 0), (0, 0))
        self.environment.current_state = state

        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(((2, 0), (1, 0)), next_state)

        ################################################################################################################
        # Set to (1, 0) and go to get gold, but there isn't
        ################################################################################################################
        self.environment.reset()

        state = ((1, 0), (0, 0))
        self.environment.current_state = state

        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual(((2, 0), (1, 0)), next_state)

        ################################################################################################################
        # Set to (4, 2) and go to get gem
        ################################################################################################################
        self.environment.reset()

        state = ((4, 2), (0, 0))
        self.environment.current_state = state

        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(((4, 1), (0, 1)), next_state)

        ################################################################################################################
        # Set to (4, 2) and go to get gem, but there isn't
        ################################################################################################################
        self.environment.reset()

        state = ((4, 2), (0, 0))
        self.environment.current_state = state

        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual(((4, 1), (0, 1)), next_state)

    def test_action_space_length(self):
        self.assertEqual(4, self.environment.action_space.n)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Simple valid step
        # Reward:
        #   [enemy_attack, gold, gems]
        # Complex position:
        #   (position, resources_available)
        # Remember that initial position is (2, 4)

        # Disable enemy attack
        self.environment.p_attack = 0

        next_state, reward, is_final = None, None, None

        # Do 2 steps to RIGHT
        for _ in range(2):
            _ = self.environment.step(action=self.environment.actions['RIGHT'])

        # Do 3 steps to UP (Get a gem)
        for _ in range(3):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual(((4, 1), (0, 1)), next_state)
        self.assertEqual([0, 0, 0], reward)
        self.assertFalse(is_final)

        _ = self.environment.step(action=self.environment.actions['UP'])

        # Do 2 steps to LEFT
        for _ in range(2):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['LEFT'])

        self.assertEqual(((2, 0), (1, 1)), next_state)
        self.assertEqual([0, 0, 0], reward)
        self.assertFalse(is_final)

        # Go to home
        # Do 4 steps to DOWN
        for _ in range(4):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        self.assertEqual(((2, 4), (1, 1)), next_state)
        self.assertEqual([0, 1, 1], reward)
        self.assertFalse(is_final)

        ################################################################################################################
        # Trying get gold through enemy
        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Do 4 steps to UP
        for _ in range(4):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['UP'])

        self.assertEqual(((2, 0), (1, 0)), next_state)
        self.assertEqual([0, 0, 0], reward)
        self.assertFalse(is_final)

        # Force to enemy attack
        self.environment.p_attack = 1

        # Go to enemy position
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['DOWN'])

        # Reset at home
        self.assertEqual(((2, 4), (0, 0)), next_state)
        self.assertEqual([-1, 0, 0], reward)
        self.assertFalse(is_final)

    def test_states_size(self):
        self.assertEqual(93, len(self.environment.states()))

    def test_transition_reward(self):

        # In this environment doesn't mind initial state to get the reward
        for state in self.environment.states():

            self.environment.current_state = state

            # Doesn't mind action too.
            for a in self.environment.action_space:

                for reachable_state in self.environment.reachable_states(state=state, action=a):

                    # Decompose next state
                    next_position, next_objects = reachable_state

                    reward = self.environment.transition_reward(state=state, action=a, next_state=reachable_state)

                    # Reach any final state
                    if reachable_state in {
                        ((2, 4), (0, 1)), ((2, 4), (1, 0)), ((2, 4), (1, 1))
                    }:
                        expected_reward = list(next_objects)
                        expected_reward.insert(0, 0)

                        self.assertEqual(expected_reward, reward)

                    # It'state attacked
                    elif self.environment.warning_action(state=state, action=a) and next_position == (2, 4):
                        self.assertEqual([-1, 0, 0], reward)

                    # Default reward
                    else:
                        self.assertEqual([0, 0, 0], reward)

    def test_reachable_states(self):

        limit_x = (self.environment.observation_space[0][0].n - 1)
        limit_y = (self.environment.observation_space[0][1].n - 1)

        # For any state the following happens
        for state in self.environment.states():

            # Decompose state
            position, objects = state

            # Decompose elements
            (x, y) = position
            (gold, gems) = objects

            # Go to UP
            reachable_states = self.environment.reachable_states(state=state, action=self.environment.actions['UP'])
            reachable_states_len = len(reachable_states)

            expected_reachable_states = set()
            expected_reachable_states_len = 1

            if position == (2, 2):
                expected_reachable_states_len = 2
                expected_reachable_states.add(((2, 4), (0, 0)))
                expected_reachable_states.add(((2, 1), objects))

            elif position == (3, 1) or position == (3, 0):
                expected_reachable_states_len = 2
                expected_reachable_states.add(((2, 4), (0, 0)))
                expected_reachable_states.add(((3, 0), objects))

            elif position == (4, 2):
                expected_reachable_states.add(((x, y - 1), (gold, 1)))

            elif position == (2, 1):
                expected_reachable_states.add(((x, y - 1), (1, gems)))

            elif y <= 0:
                expected_reachable_states.add(((x, y), objects))

            elif y > 0:
                expected_reachable_states.add(((x, y - 1), objects))

            self.assertEqual(expected_reachable_states_len, reachable_states_len)
            self.assertTrue(
                all(
                    element in expected_reachable_states for element in reachable_states
                ) and
                all(
                    element in reachable_states for element in expected_reachable_states
                )
            )

            # Go to RIGHT
            reachable_states = self.environment.reachable_states(state=state, action=self.environment.actions['RIGHT'])
            reachable_states_len = len(reachable_states)

            expected_reachable_states = set()
            expected_reachable_states_len = 1

            if position == (2, 0):
                expected_reachable_states_len = 2
                expected_reachable_states.add(((2, 4), (0, 0)))
                expected_reachable_states.add(((3, 0), objects))

            elif position == (1, 1):
                expected_reachable_states_len = 2
                expected_reachable_states.add(((2, 4), (0, 0)))
                expected_reachable_states.add(((2, 1), objects))

            elif position == (3, 1):
                expected_reachable_states.add(((x + 1, y), (gold, 1)))

            elif position == (1, 0):
                expected_reachable_states.add(((x + 1, y), (1, gems)))

            elif x >= limit_x:
                expected_reachable_states.add(((x, y), objects))

            elif x < limit_x:
                expected_reachable_states.add(((x + 1, y), objects))

            self.assertEqual(expected_reachable_states_len, reachable_states_len)
            self.assertTrue(
                all(
                    element in expected_reachable_states for element in reachable_states
                ) and
                all(
                    element in reachable_states for element in expected_reachable_states
                )
            )

            # Go to DOWN
            reachable_states = self.environment.reachable_states(state=state, action=self.environment.actions['DOWN'])
            reachable_states_len = len(reachable_states)

            expected_reachable_states = set()
            expected_reachable_states_len = 1

            if position == (2, 0):
                expected_reachable_states_len = 2
                expected_reachable_states.add(((2, 4), (0, 0)))
                expected_reachable_states.add(((2, 1), objects))

            elif position == (4, 0):
                expected_reachable_states.add(((x, y + 1), (gold, 1)))

            elif y >= limit_y:
                expected_reachable_states.add(((x, y), objects))

            elif y < limit_y:
                expected_reachable_states.add(((x, y + 1), objects))

            self.assertEqual(expected_reachable_states_len, reachable_states_len)
            self.assertTrue(
                all(
                    element in expected_reachable_states for element in reachable_states
                ) and
                all(
                    element in reachable_states for element in expected_reachable_states
                )
            )

            # Go to LEFT
            reachable_states = self.environment.reachable_states(state=state, action=self.environment.actions['LEFT'])
            reachable_states_len = len(reachable_states)

            expected_reachable_states = set()
            expected_reachable_states_len = 1

            if position == (4, 0):
                expected_reachable_states_len = 2
                expected_reachable_states.add(((2, 4), (0, 0)))
                expected_reachable_states.add(((3, 0), objects))

            elif position == (3, 1):
                expected_reachable_states_len = 2
                expected_reachable_states.add(((2, 4), (0, 0)))
                expected_reachable_states.add(((2, 1), objects))

            elif position == (3, 0):
                expected_reachable_states.add(((x - 1, y), (1, gems)))

            elif x <= 0:
                expected_reachable_states.add(((x, y), objects))

            elif x > 0:
                expected_reachable_states.add(((x - 1, y), objects))

            self.assertEqual(expected_reachable_states_len, reachable_states_len)
            self.assertTrue(
                all(
                    element in expected_reachable_states for element in reachable_states
                ) and
                all(
                    element in reachable_states for element in expected_reachable_states
                )
            )

    def test_transition_probability(self):

        for state in self.environment.states():

            self.environment.current_state = state

            for a in self.environment.action_space:

                for reachable_state in self.environment.reachable_states(state=state, action=a):

                    probability = self.environment.transition_probability(
                        state=state, action=a, next_state=reachable_state
                    )

                    if self.environment.warning_action(state=state, action=a):

                        if reachable_state[0] == (2, 4):
                            self.assertEqual(self.environment.p_attack, probability)
                        else:
                            self.assertEqual(1. - self.environment.p_attack, probability)

                    else:
                        self.assertEqual(1., probability)
