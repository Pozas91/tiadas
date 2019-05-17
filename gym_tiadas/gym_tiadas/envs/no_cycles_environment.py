"""
Environment to test Lawrence and José Luis paper Algorithm 1.
"""
import gym

from models import Vector
from spaces import DynamicSpace
from .environment import Environment


class NoCyclesEnvironment(Environment):
    # Possible actions
    _actions = {'a1': 0, 'a2': 1, 'a3': 2, 'a4': 3, 'a5': 4, 'a6': 5, 'a7': 6, 'a8': 7, 'a9': 8}

    # Icons to render environments
    _icons = {}

    def __init__(self, seed: int = 0, initial_state: int = 0, default_reward: tuple = (-1, 0)):
        """
        :param seed:
        :param initial_state:
        """

        # Create the observation space
        observation_space = gym.spaces.Discrete(7)

        # Default reward
        default_reward = Vector(default_reward)

        # Super call constructor
        super().__init__(observation_space=observation_space, seed=seed, initial_state=initial_state,
                         default_reward=default_reward)

        # Available transitions
        self.transitions = {
            0: {
                self._actions.get('a1'): [
                    1, 2
                ],
                self._actions.get('a2'): [
                    2, 3
                ]
            },
            1: {
                self._actions.get('a3'): [
                    4
                ]
            },
            2: {
                self._actions.get('a5'): [
                    4, 5
                ],
                self._actions.get('a6'): [
                    3
                ]
            },
            3: {
                self._actions.get('a7'): [
                    5
                ]
            },
            5: {
                self._actions.get('a8'): [
                    6
                ],
                self._actions.get('a9'): [
                    7
                ]
            }
        }

        # Rewards
        self.rewards = {
            4: Vector([0, 3]),
            6: Vector([0, 8]),
            7: Vector([0, 20])
        }

    def step(self, action: int) -> (int, Vector, bool, dict):
        """
        Take a step in the environment
        :param action:
        :return:
        """

        # Get new state
        new_state = self.next_state(action=action)

        # Get reward
        reward = self.rewards.get(new_state, self.default_reward)

        # Update previous state
        self.current_state = new_state

        # Check is_final
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
        possible_states = self.transitions.get(state).get(action)

        # Return new state
        return self.np_random.choice(possible_states)

    def is_final(self, state: int = None) -> bool:
        """
        Checks if this is final state. 
        :param state:
        :return:
        """

        # If a state is given get that state, else get current_state
        state = self.current_state if state is None else state

        # Check if state is in rewards keys.
        return state in self.rewards.keys()

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment data
        del data['transitions']
        del data['rewards']

        return data

    @property
    def action_space(self) -> DynamicSpace:
        # Get current state
        state = self.current_state

        # Get all actions available
        keys = self.transitions.get(state, {}).keys()

        # Return complete valid_actions
        valid_actions = DynamicSpace([value for value in self._action_space if value in keys])

        # Return and list of iterable valid actions
        return valid_actions
