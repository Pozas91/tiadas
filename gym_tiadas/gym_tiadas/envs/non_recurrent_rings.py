"""
Such as LinkedRings environment, but in this case we have an extra state and different values.
"""
import gym

from models import Vector
from .environment import Environment


class NonRecurrentRings(Environment):
    # Possible actions
    _actions = {'CLOCKWISE': 0, 'COUNTER-CLOCKWISE': 1}

    # Icons to render environments
    _icons = {'BLANK': ' ', 'BLOCK': '■', 'TREASURE': '$', 'CURRENT': '☺', 'ENEMY': '×', 'HOME': 'µ', 'FINAL': '$'}

    def __init__(self, seed: int = 0, initial_state: int = 0, default_reward: tuple = (0, 0)):
        """
        :param seed:
        :param initial_state:
        """

        # Create the observation space
        observation_space = gym.spaces.Discrete(8)

        super().__init__(observation_space=observation_space, seed=seed, initial_state=initial_state)

        # Rewards dictionary
        self.rewards_dictionary = {
            0: {
                self.actions.get('COUNTER-CLOCKWISE'): Vector([2, -1]),
                self.actions.get('CLOCKWISE'): Vector([-1, 0])
            },
            1: {
                self.actions.get('COUNTER-CLOCKWISE'): Vector([2, -1]),
                self.actions.get('CLOCKWISE'): Vector([-1, 0])
            },
            2: {
                self.actions.get('COUNTER-CLOCKWISE'): Vector([2, -1]),
                self.actions.get('CLOCKWISE'): Vector([-1, 0])
            },
            3: {
                self.actions.get('COUNTER-CLOCKWISE'): Vector([2, -1]),
                self.actions.get('CLOCKWISE'): Vector([-1, 0])
            },
            4: {
                self.actions.get('CLOCKWISE'): Vector([-1, 2]),
                self.actions.get('COUNTER-CLOCKWISE'): Vector([0, -1])
            },
            5: {
                self.actions.get('CLOCKWISE'): Vector([-1, 2]),
                self.actions.get('COUNTER-CLOCKWISE'): Vector([0, -1])
            },
            6: {
                self.actions.get('CLOCKWISE'): Vector([-1, 2]),
                self.actions.get('COUNTER-CLOCKWISE'): Vector([0, -1])
            },
            7: {
                self.actions.get('CLOCKWISE'): Vector([-1, 2]),
                self.actions.get('COUNTER-CLOCKWISE'): Vector([0, -1])
            }
        }

        # Possible transitions from a state to another
        self.possible_transitions = {
            0: {
                self.actions.get('COUNTER-CLOCKWISE'): 1,
                self.actions.get('CLOCKWISE'): 7
            },
            1: {
                self.actions.get('COUNTER-CLOCKWISE'): 2,
                self.actions.get('CLOCKWISE'): 0
            },
            2: {
                self.actions.get('COUNTER-CLOCKWISE'): 3,
                self.actions.get('CLOCKWISE'): 1
            },
            3: {
                self.actions.get('COUNTER-CLOCKWISE'): 0,
                self.actions.get('CLOCKWISE'): 2
            },
            4: {
                self.actions.get('CLOCKWISE'): 5,
                self.actions.get('COUNTER-CLOCKWISE'): 7
            },
            5: {
                self.actions.get('CLOCKWISE'): 6,
                self.actions.get('COUNTER-CLOCKWISE'): 4
            },
            6: {
                self.actions.get('CLOCKWISE'): 7,
                self.actions.get('COUNTER-CLOCKWISE'): 5
            },
            7: {
                self.actions.get('CLOCKWISE'): 4,
                self.actions.get('COUNTER-CLOCKWISE'): 0
            }
        }

        self.default_reward = Vector(default_reward)

    def step(self, action: int) -> (int, Vector, bool, dict):
        """
        Do a step in the environment
        :param action:
        :return:
        """

        # Get new state
        new_state = self.next_state(action=action)

        # Get reward
        reward = self.rewards_dictionary.get(self.current_state).get(action)

        # Update previous state
        self.current_state = new_state

        # Check if is final state
        final = self.is_final()

        # Set info
        info = {}

        return new_state, reward, final, info

    def reset(self) -> int:
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        return self.current_state

    def next_state(self, action: int, state: int = None) -> int:
        """
        Calc next state with state and action given.
        :param state: if a state is given, process next_state from that state, else get current state.
        :param action: from action_space
        :return: a new state (or old if is invalid action)
        """

        # Check if a state is given.
        state = self.current_state if state is None else state

        # Do movement
        new_state = self.possible_transitions.get(state).get(action)

        if not self.observation_space.contains(new_state):
            # New state is invalid, and roll back with previous.
            new_state = self.current_state

        # Return new state
        return new_state

    def is_final(self, state: int = None) -> bool:
        # (This is a non-episodic problem, so doesn't have final states)
        return False

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment data
        del data['possible_transitions']
        del data['rewards_dictionary']

        return data
