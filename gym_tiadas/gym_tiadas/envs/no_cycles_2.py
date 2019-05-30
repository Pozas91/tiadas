"""
Environment to test Lawrence and José Luis paper Algorithm 1.
"""
import gym

from models import Vector
from spaces import DynamicSpace
from .environment import Environment


class NoCycles2(Environment):
    # Possible actions
    _actions = {'a0': 0, 'a1': 1, 'a2': 2, 'a3': 3, 'a4': 4}

    # Icons to render environments
    _icons = {}

    def __init__(self, seed: int = 0, initial_state: int = 0, default_reward: tuple = (-1, 0)):
        """
        :param seed:
        :param initial_state:
        """

        # Create the observation space
        observation_space = gym.spaces.Discrete(5)

        # Default reward
        default_reward = Vector(default_reward)

        # Super call constructor
        super().__init__(observation_space=observation_space, seed=seed, initial_state=initial_state,
                         default_reward=default_reward)

        # Available transitions
        self.transitions = {
            0: {
                self._actions.get('a0'): [
                    1, 2
                ]
            },
            1: {
                self._actions.get('a1'): [
                    3
                ],
                self._actions.get('a2'): [
                    4
                ],
            },
            2: {
                self._actions.get('a3'): [
                    3
                ],
                self._actions.get('a4'): [
                    4
                ],
            }
        }

        # Rewards
        self.rewards = {
            3: Vector([0, 4]),
            4: Vector([-1, 5]),
        }

        # Trying improve performance
        self.dynamic_action_space = DynamicSpace([])

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

        # Getting possible actions to transition
        possible_actions = [value for value in self._action_space if value in keys]

        # Setting to dynamic_space
        self.dynamic_action_space.items = possible_actions

        # Update n length
        self.dynamic_action_space.n = len(possible_actions)

        # Return a list of iterable valid actions
        return self.dynamic_action_space
