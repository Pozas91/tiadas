"""
Such as LinkedRings environment, but in this case we have an extra position and different values.
"""
import gym

from models import Vector
from .environment import Environment


class NonRecurrentRings(Environment):
    # Possible actions
    _actions = {'CLOCKWISE': 0, 'COUNTER-CLOCKWISE': 1}

    def __init__(self, seed: int = 0, initial_state: int = 0, default_reward: tuple = (0, 0)):
        """
        :param seed:
        :param initial_state:
        """

        # Create the observation space
        observation_space = gym.spaces.Discrete(8)

        # Default reward
        default_reward = Vector(default_reward)

        super().__init__(observation_space=observation_space, seed=seed, initial_state=initial_state,
                         default_reward=default_reward)

        # Rewards dictionary
        self.rewards_dictionary = {
            0: {
                self.actions['COUNTER-CLOCKWISE']: Vector([2, -1]),
                self.actions['CLOCKWISE']: Vector([-1, 0])
            },
            1: {
                self.actions['COUNTER-CLOCKWISE']: Vector([2, -1]),
                self.actions['CLOCKWISE']: Vector([-1, 0])
            },
            2: {
                self.actions['COUNTER-CLOCKWISE']: Vector([2, -1]),
                self.actions['CLOCKWISE']: Vector([-1, 0])
            },
            3: {
                self.actions['COUNTER-CLOCKWISE']: Vector([2, -1]),
                self.actions['CLOCKWISE']: Vector([-1, 0])
            },
            4: {
                self.actions['CLOCKWISE']: Vector([-1, 2]),
                self.actions['COUNTER-CLOCKWISE']: Vector([0, -1])
            },
            5: {
                self.actions['CLOCKWISE']: Vector([-1, 2]),
                self.actions['COUNTER-CLOCKWISE']: Vector([0, -1])
            },
            6: {
                self.actions['CLOCKWISE']: Vector([-1, 2]),
                self.actions['COUNTER-CLOCKWISE']: Vector([0, -1])
            },
            7: {
                self.actions['CLOCKWISE']: Vector([-1, 2]),
                self.actions['COUNTER-CLOCKWISE']: Vector([0, -1])
            }
        }

        # Possible p_stochastic from a position to another
        self.possible_transitions = {
            0: {
                self.actions['COUNTER-CLOCKWISE']: 1,
                self.actions['CLOCKWISE']: 7
            },
            1: {
                self.actions['COUNTER-CLOCKWISE']: 2,
                self.actions['CLOCKWISE']: 0
            },
            2: {
                self.actions['COUNTER-CLOCKWISE']: 3,
                self.actions['CLOCKWISE']: 1
            },
            3: {
                self.actions['COUNTER-CLOCKWISE']: 0,
                self.actions['CLOCKWISE']: 2
            },
            4: {
                self.actions['CLOCKWISE']: 5,
                self.actions['COUNTER-CLOCKWISE']: 7
            },
            5: {
                self.actions['CLOCKWISE']: 6,
                self.actions['COUNTER-CLOCKWISE']: 4
            },
            6: {
                self.actions['CLOCKWISE']: 7,
                self.actions['COUNTER-CLOCKWISE']: 5
            },
            7: {
                self.actions['CLOCKWISE']: 4,
                self.actions['COUNTER-CLOCKWISE']: 0
            }
        }

    def step(self, action: int) -> (int, Vector, bool, dict):
        """
        Do a step in the environment
        :param action:
        :return:
        """

        # Get next position
        next_state = self.next_state(action=action)

        # Get reward
        reward = self.rewards_dictionary[self.current_state][action]

        # Update previous position
        self.current_state = next_state

        # Check if is final position
        final = self.is_final()

        # Set extra
        info = {}

        return next_state, reward, final, info

    def next_state(self, action: int, state: int = None) -> int:
        """
        Calc next position with position and action given.
        :param state: if a position is given, process next_state from that position, else get current position.
        :param action: from action_space
        :return: a new position (or old if is invalid action)
        """

        # Check if a position is given.
        position = state if state else self.current_state

        # Do movement
        next_position = self.possible_transitions[position][action]

        if not self.observation_space.contains(next_position):
            # New position is invalid, and roll back with previous.
            next_position = position

        # Return new position
        return next_position

    def is_final(self, state: int = None) -> bool:
        # (This is a non-episodic problem, so doesn't have final states)
        return False

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment train_data
        del data['possible_transitions']
        del data['rewards_dictionary']

        return data

    def transition_reward(self, state: int, action: int, next_state: int) -> Vector:
        return self.rewards_dictionary[state][action]

    def states(self) -> set:
        return set(range(self.observation_space.n))

    def reachable_states(self, state: int, action: int) -> set:
        return {self.possible_transitions[state][action]}
