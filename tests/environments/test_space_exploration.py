"""
Unit tests file where testing SpaceExploration environment.
"""

from gym import spaces

import utils.environments as ue
from environments import SpaceExploration
from models import Vector
from tests.environments.test_env_mesh import TestEnvMesh


class TestSpaceExploration(TestEnvMesh):

    def setUp(self):
        # Set seed to 0 to testing.
        self.environment = SpaceExploration(seed=0)

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        super().test_init()

        # This environment must have another attributes
        self.assertTrue(hasattr(self.environment, 'radiations'))
        self.assertTrue(hasattr(self.environment, 'asteroids'))

        # By default mesh shape is 13x5
        self.assertEqual(spaces.Tuple((spaces.Discrete(13), spaces.Discrete(5))), self.environment.observation_space)

        # Default reward is (0, -1)
        self.assertEqual((0, -1), self.environment.default_reward)

    def test_action_space_length(self):
        self.assertEqual(len(self.environment.actions), 8)

    def test_initial_state(self):
        # By default initial position is (5, 2)
        self.assertEqual((5, 2), self.environment.initial_state)

    def test__next_state(self):
        """
        Testing _next_state method
        :return:
        """

        # THIS IS A CYCLIC ENVIRONMENT

        ################################################################################################################
        # Begin at position (0, 0) (TOP-LEFT corner)
        ################################################################################################################
        state = (0, 0)
        self.environment.current_state = state

        # Go to UP
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual((0, 4), next_state)

        # Go to UP RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['UP RIGHT'])
        self.assertEqual((1, 4), next_state)

        # Go to RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((1, 0), next_state)

        # Go to DOWN RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN RIGHT'])
        self.assertEqual((1, 1), next_state)

        # Go to DOWN
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((0, 1), next_state)

        # Go to DOWN LEFT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN LEFT'])
        self.assertEqual((12, 1), next_state)

        # Go to LEFT
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual((12, 0), next_state)

        # Go to UP LEFT
        next_state = self.environment.next_state(action=self.environment.actions['UP LEFT'])
        self.assertEqual((12, 4), next_state)

        ################################################################################################################
        # Begin at position (12, 0) (TOP-RIGHT corner)
        ################################################################################################################
        state = (12, 0)
        self.environment.current_state = state

        # Go to UP
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual((12, 4), next_state)

        # Go to UP RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['UP RIGHT'])
        self.assertEqual((0, 4), next_state)

        # Go to RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((0, 0), next_state)

        # Go to DOWN RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN RIGHT'])
        self.assertEqual((0, 1), next_state)

        # Go to DOWN
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((12, 1), next_state)

        # Go to DOWN LEFT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN LEFT'])
        self.assertEqual((11, 1), next_state)

        # Go to LEFT
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual((11, 0), next_state)

        # Go to UP LEFT
        next_state = self.environment.next_state(action=self.environment.actions['UP LEFT'])
        self.assertEqual((11, 4), next_state)

        ################################################################################################################
        # Begin at position (12, 4) (DOWN-RIGHT corner)
        ################################################################################################################
        state = (12, 4)
        self.environment.current_state = state

        # Go to UP
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual((12, 3), next_state)

        # Go to UP RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['UP RIGHT'])
        self.assertEqual((0, 3), next_state)

        # Go to RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((0, 4), next_state)

        # Go to DOWN RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN RIGHT'])
        self.assertEqual((0, 0), next_state)

        # Go to DOWN
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((12, 0), next_state)

        # Go to DOWN LEFT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN LEFT'])
        self.assertEqual((11, 0), next_state)

        # Go to LEFT
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual((11, 4), next_state)

        # Go to UP LEFT
        next_state = self.environment.next_state(action=self.environment.actions['UP LEFT'])
        self.assertEqual((11, 3), next_state)

        ################################################################################################################
        # Begin at position (0, 4) (DOWN-LEFT corner)
        ################################################################################################################
        state = (0, 4)
        self.environment.current_state = state

        # Go to UP
        next_state = self.environment.next_state(action=self.environment.actions['UP'])
        self.assertEqual((0, 3), next_state)

        # Go to UP RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['UP RIGHT'])
        self.assertEqual((1, 3), next_state)

        # Go to RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['RIGHT'])
        self.assertEqual((1, 4), next_state)

        # Go to DOWN RIGHT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN RIGHT'])
        self.assertEqual((1, 0), next_state)

        # Go to DOWN
        next_state = self.environment.next_state(action=self.environment.actions['DOWN'])
        self.assertEqual((0, 0), next_state)

        # Go to DOWN LEFT
        next_state = self.environment.next_state(action=self.environment.actions['DOWN LEFT'])
        self.assertEqual((12, 0), next_state)

        # Go to LEFT
        next_state = self.environment.next_state(action=self.environment.actions['LEFT'])
        self.assertEqual((12, 4), next_state)

        # Go to UP LEFT
        next_state = self.environment.next_state(action=self.environment.actions['UP LEFT'])
        self.assertEqual((12, 3), next_state)

    def test_step(self):
        """
        Testing step method
        :return:
        """

        # Reward:
        #   [mission_success, radiation]
        # Remember that initial position is (5, 2)

        # Reset environment
        self.environment.reset()

        next_state, reward, is_final = None, None, None

        # Go to RIGHT 2 steps, until (5, 4), ship is destroyed by asteroid.
        for _ in range(2):
            next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['RIGHT'])

        self.assertEqual((7, 2), next_state)
        self.assertEqual([-100, -1], reward)
        self.assertTrue(is_final)

        ################################################################################################################

        # Reset environment
        self.environment.reset()

        # Go to a final position.
        # (4, 2)
        _ = self.environment.step(action=self.environment.actions['LEFT'])

        # (3, 1)
        _ = self.environment.step(action=self.environment.actions['UP LEFT'])

        # (2, 1)
        _ = self.environment.step(action=self.environment.actions['LEFT'])

        # (1, 0), position with radiation
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['UP LEFT'])

        self.assertEqual((1, 0), next_state)
        self.assertEqual([0, -11], reward)
        self.assertFalse(is_final)

        # (0, 4)
        next_state, reward, is_final, _ = self.environment.step(action=self.environment.actions['UP LEFT'])

        self.assertEqual((0, 4), next_state)
        self.assertEqual([20, -1], reward)
        self.assertTrue(is_final)

    def test_states_size(self):
        self.assertEqual(len(self.environment.states()), 52)

    def test_transition_reward(self):

        # In this environment doesn't mind initial state to get the reward
        state = self.environment.observation_space.sample()

        # Doesn't mind action too.
        action = self.environment.action_space.sample()

        # Asteroids states
        for asteroid_state in self.environment.asteroids:
            self.assertEqual(
                Vector((-100, -1)),
                self.environment.transition_reward(
                    state=state, action=action, next_state=asteroid_state
                )
            )

        # Radiations states
        for radiation_state in self.environment.radiations:
            self.assertEqual(
                Vector((0, -11)),
                self.environment.transition_reward(
                    state=state, action=action, next_state=radiation_state
                )
            )

        # Finals states
        for final_state, final_reward in self.environment.finals.items():
            self.assertEqual(
                Vector((final_reward, -1)),
                self.environment.transition_reward(
                    state=state, action=action, next_state=final_state
                )
            )

        simple_states = self.environment.states() - set(self.environment.finals.keys()).union(
            self.environment.radiations).union(self.environment.asteroids)

        for simple_state in simple_states:
            self.assertEqual(
                Vector((0, -1)),
                self.environment.transition_reward(
                    state=state, action=action, next_state=simple_state
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

        # For any state the following happens
        for state in self.environment.states():
            # Decompose state
            x, y = state

            ############################################################################################################
            # Go to up
            reachable_states = self.environment.reachable_states(state=state, action=self.environment.actions['UP'])

            x_moved = x
            y_moved = ue.move_up(y=y, limit=self.environment.observation_space[1].n)

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x_moved, y_moved)})
            ############################################################################################################
            # Go to up-right
            reachable_states = self.environment.reachable_states(
                state=state, action=self.environment.actions['UP RIGHT']
            )

            x_moved = ue.move_right(x=x, limit=self.environment.observation_space[0].n)
            y_moved = ue.move_up(y=y, limit=self.environment.observation_space[1].n)

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x_moved, y_moved)})
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
            ############################################################################################################
            # Go to down-left
            reachable_states = self.environment.reachable_states(
                state=state, action=self.environment.actions['DOWN LEFT']
            )

            x_moved = ue.move_left(x=x, limit=self.environment.observation_space[0].n)
            y_moved = ue.move_down(y=y, limit=self.environment.observation_space[1].n)

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x_moved, y_moved)})
            ############################################################################################################
            # Go to left
            reachable_states = self.environment.reachable_states(
                state=state, action=self.environment.actions['LEFT']
            )

            x_moved = ue.move_left(x=x, limit=self.environment.observation_space[0].n)
            y_moved = y

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x_moved, y_moved)})
            ############################################################################################################
            # Go to up-left
            reachable_states = self.environment.reachable_states(
                state=state, action=self.environment.actions['UP LEFT']
            )

            x_moved = ue.move_left(x=x, limit=self.environment.observation_space[0].n)
            y_moved = ue.move_up(y=y, limit=self.environment.observation_space[1].n)

            self.assertTrue(len(reachable_states) == 1)
            self.assertIn(next(iter(reachable_states)), {(x_moved, y_moved)})
            ############################################################################################################
