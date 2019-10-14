"""
Base class to define environments with an observation_space defined by a mesh.
Mesh is a grid of columns per rows, where the minimum value is (0, 0), and maximum 
value is (columns - 1, rows - 1).
The top-left corner is (0, 0), the top-right corner is (columns - 1, 0), bottom-left 
corner is (0, rows - 1) and bottom-right corner is (columns - 1, rows - 1).

Action space is a discrete number. Actions are in the range [0, n).

Finals is a dictionary where the key is a state, and the value is a reward
vector as follows:
{
    state_1: reward,
    state_2: reward,
    ...
}

Obstacles is a list of states: [state_1, state_2, ...]
"""

import gym
from gym import spaces

from models import Vector
from .environment import Environment


class EnvMesh(Environment):

    def __init__(self, mesh_shape: tuple, default_reward: Vector, seed: int = None, initial_state: tuple = None,
                 obstacles: frozenset = None, finals: dict = None):

        """
        :param mesh_shape: A tuple where first component represents the x-axis
                           (i.e. columns), and the second component the y-axis
                           (i.e. rows).
        :param default_reward: Default reward returned by the environment when a 
                               reward is not defined.
        :param seed: Initial seed for the random number generator.
        :param initial_state: start state for each episode.
        :param obstacles: inaccessible states.
        :param finals: terminal states for episodes.
        """

        # Create the mesh
        x, y = mesh_shape
        observation_space = gym.spaces.Tuple((spaces.Discrete(x), spaces.Discrete(y)))

        super().__init__(observation_space=observation_space, default_reward=default_reward, seed=seed,
                         initial_state=initial_state, obstacles=obstacles, finals=finals)

    def render(self, mode: str = 'human') -> None:
        """
        Render the environment as a grid on the screen
        :param mode:
        :return:
        """

        if mode == 'human':
            # Get cols (x) and rows (y) from the observation space
            cols, rows = self.observation_space.spaces[0].n, self.observation_space.spaces[1].n

            for y in range(rows):
                for x in range(cols):

                    # Set a state
                    state = (x, y)

                    if state == self.current_state:
                        icon = self._icons.get('CURRENT')
                    elif state in self.obstacles:
                        icon = self._icons.get('BLOCK')
                    elif state in self.finals.keys():
                        icon = self.finals.get(state)
                    else:
                        icon = self._icons.get('BLANK')

                    # Show col
                    print('| {} '.format(icon), end='')

                # New row
                print('|')

            # End render
            print('')

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next state with current state and action given. 
        Default is 4-neighbors (UP, LEFT, DOWN, RIGHT)
        :param state: If a state is given, do action from that state.
        :param action: from action_space
        :return: a new state (or old if is invalid action)
        """

        # Get my position
        x, y = state if state else self.current_state

        # Do movement
        if action == self._actions.get('UP'):
            y -= 1
        elif action == self._actions.get('RIGHT'):
            x += 1
        elif action == self._actions.get('DOWN'):
            y += 1
        elif action == self._actions.get('LEFT'):
            x -= 1

        # Set new state
        new_state = x, y

        # If exists obstacles, then new_state must be in self.obstacles
        is_obstacle = bool(self.obstacles) and new_state in self.obstacles

        if not self.observation_space.contains(new_state) or is_obstacle:
            # New state is invalid.
            new_state = self.current_state

        # Return (x, y) position
        return new_state

    def states(self) -> set:
        """
        Return all possible states of this environment.
        :return:
        """

        # Unpack spaces
        x_space, y_space = self.observation_space.spaces

        # Return all spaces
        return {(x, y) for x in range(x_space.n) for y in range(y_space.n)} - self.obstacles

    def reachable_states(self, state: tuple, action: int) -> set:
        return {self.next_state(action=action, state=state)}
