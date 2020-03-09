"""
Simple deterministic bi-objective non-episodic environment.

STATE SPACE:
-----------
The space position consists of 7 states linked by actions in a two ring shape.
State 1 is common to both rings.

     S2      S5
   /    \  /   \
S3       S1     S6
   \    / \     /
     S4      S7

All arcs are bidirectional, except  S4 -> S1 and S7 -> S1.
Therefore, the agent has two actions available at each position,

* Move clockwise
* Move counter-clockwise

States are implemented in a discrete 0-6 range as follows,

     1       4
   /    \  /   \
 2       0      5
   \    / \    /
     3       6


REWARDS:
-------
Left ring: (3,-1) moving counter-clockwise, (-1,0) otherwise.
Right ring: (-1,3) moving clockwise, (0, -1) otherwise.

FINAL STATE:
-----------
It is not an episodic task. Does not have finals states.

Reference:
Steering approaches to Pareto-optimal multiobjective reinforcement learning.
Peter Vamplew, Rustam Issabekov, Richard Dazeley, Cameron Foale, Adam Berry, 
Tim Moore, Douglas Creighton
Neurocomputing 263 (2017) 26-38.
"""
import gym

from models import Vector
from .environment import Environment


class LinkedRings(Environment):
    # Possible actions
    _actions = {'CLOCKWISE': 0, 'COUNTER_CLOCKWISE': 1}

    def __init__(self, seed: int = 0, initial_state: int = 0, default_reward: tuple = (0, 0)):
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

        # Rewards dictionary
        self.rewards_dictionary = {
            0: {
                self.actions['COUNTER_CLOCKWISE']: Vector([3, -1]),
                self.actions['CLOCKWISE']: Vector([-1, 3])
            },
            1: {
                self.actions['COUNTER_CLOCKWISE']: Vector([3, -1]),
                self.actions['CLOCKWISE']: Vector([-1, 0])
            },
            2: {
                self.actions['COUNTER_CLOCKWISE']: Vector([3, -1]),
                self.actions['CLOCKWISE']: Vector([-1, 0])
            },
            3: {
                self.actions['COUNTER_CLOCKWISE']: Vector([3, -1]),
                self.actions['CLOCKWISE']: Vector([-1, 0])
            },
            4: {
                self.actions['CLOCKWISE']: Vector([-1, 3]),
                self.actions['COUNTER_CLOCKWISE']: Vector([0, -1])
            },
            5: {
                self.actions['CLOCKWISE']: Vector([-1, 3]),
                self.actions['COUNTER_CLOCKWISE']: Vector([0, -1])
            },
            6: {
                self.actions['CLOCKWISE']: Vector([-1, 3]),
                self.actions['COUNTER_CLOCKWISE']: Vector([0, -1])
            }
        }

        # Possible p_stochastic from a position to another
        self.possible_transitions = {
            0: {
                self.actions['COUNTER_CLOCKWISE']: 1,
                self.actions['CLOCKWISE']: 4
            },
            1: {
                self.actions['COUNTER_CLOCKWISE']: 2,
                self.actions['CLOCKWISE']: 0
            },
            2: {
                self.actions['COUNTER_CLOCKWISE']: 3,
                self.actions['CLOCKWISE']: 1
            },
            3: {
                self.actions['COUNTER_CLOCKWISE']: 0,
                self.actions['CLOCKWISE']: 2
            },
            4: {
                self.actions['CLOCKWISE']: 5,
                self.actions['COUNTER_CLOCKWISE']: 0
            },
            5: {
                self.actions['CLOCKWISE']: 6,
                self.actions['COUNTER_CLOCKWISE']: 4
            },
            6: {
                self.actions['CLOCKWISE']: 0,
                self.actions['COUNTER_CLOCKWISE']: 5
            }
        }

    def step(self, action: int) -> (int, Vector, bool, dict):
        """
        Take a step in the environment
        :param action:
        :return:
        """

        # Get next position
        next_state = self.next_state(action=action)

        # Get reward
        reward = self.rewards_dictionary[self.current_state][action]

        # Update previous position
        self.current_state = next_state

        # Check is_final
        final = self.is_final()

        # Set extra
        info = {}

        return next_state, reward, final, info

    def next_state(self, action: int, state: int = None) -> int:
        """
        Calc next state with state and action given.
        :param state: if a state is given, process next_position from that state, else get current state.
        :param action: from action_space
        :return: a new state (or old if is invalid action)
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
        """
        Checks if this is final position.
        :param state:
        :return: Always False, since this task is not episodic.
        """
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
