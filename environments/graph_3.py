import gym

from environments import Environment
from models import Vector


class Graph3(Environment):
    # Possible actions
    _actions = {'LEFT': 0, 'RIGHT': 1}

    def __init__(self, seed: int = 0, initial_state: int = 1, default_reward: tuple = (-1, 0)):
        """
        :param seed:
        :param initial_state:
        """

        # Create the observation space
        observation_space = gym.spaces.Discrete(4)

        # Default reward
        default_reward = Vector(default_reward)

        # Super call constructor
        super().__init__(observation_space=observation_space, seed=seed, initial_state=initial_state,
                         default_reward=default_reward)

        # Rewards dictionary
        self.rewards_dictionary = {
            0: Vector((-1, 0)),
            1: Vector((-1, 0)),
            2: Vector((-1, 0)),
            3: Vector((-1, 10))
        }

        # Possible p_stochastic from a position to another
        self.possible_transitions = {
            1: {
                self.actions['LEFT']: [0, 1],
                self.actions['RIGHT']: [2, 1]
            },
            2: {
                self.actions['LEFT']: [1, 2],
                self.actions['RIGHT']: [3, 2]
            },
        }

        self.finals = {
            0: Vector((-1, 0)),
            3: Vector((-1, 10))
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
        reward = self.rewards_dictionary[next_state]

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

        # Calc probability
        prob = self.np_random.uniform()

        # Do movement (if prob > 0.8, then True whose value is 1, that is keep in position)
        next_position = self.possible_transitions[position][action][prob > 0.8]

        if not self.observation_space.contains(next_position):
            # New position is invalid, and roll back with previous.
            next_position = position

        # Return new position
        return next_position

    def is_final(self, state: int = None) -> bool:
        return state in self.finals

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
        return self.rewards_dictionary[next_state]

    def states(self) -> set:
        return set(range(self.observation_space.n)) - set(self.finals.keys())

    def reachable_states(self, state: int, action: int) -> set:
        return set(self.possible_transitions[state][action])

    def transition_probability(self, state: int, action: int, next_state: int) -> float:
        index = self.possible_transitions[state][action].index(next_state)

        return 0.8 if index == 0 else 0.2
