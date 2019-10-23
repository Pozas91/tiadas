"""
Base class to define environments with an observation_space defined by a mesh.
Mesh is a grid of columns per rows, where the minimum value is (0, 0), and maximum 
value is (columns - 1, rows - 1).
The top-left corner is (0, 0), the top-right corner is (columns - 1, 0), bottom-left 
corner is (0, rows - 1) and bottom-right corner is (columns - 1, rows - 1).

Action space is a discrete number. Actions are in the range [0, n).

Finals is a dictionary where the key is a position, and the value is a reward
vector as follows:
{
    state_1: reward,
    state_2: reward,
    ...
}

Obstacles is a list of states: [state_1, state_2, ...]
"""
from typing import Iterable

import gym
from gym import spaces

from models import Vector
from .environment import Environment


class EnvMesh(Environment):

    def __init__(self, mesh_shape: tuple, default_reward: Vector, seed: int = None, initial_state: tuple = None,
                 obstacles: frozenset = None, finals: Iterable = None, observation_space: gym.spaces = None,
                 action_space: gym.spaces = None):

        """
        :param mesh_shape: A tuple where first component represents the x-axis (i.e. columns), and the second component
            the y-axis (i.e. rows).
        :param default_reward: Default reward returned by the environment when a  reward is not defined.
        :param seed: Initial seed for the random number generator.
        :param initial_state: start position for each episode.
        :param obstacles: inaccessible states.
        :param finals: terminal states for episodes.
        """

        if observation_space is None:
            # Create the mesh
            x, y = mesh_shape
            observation_space = gym.spaces.Tuple((spaces.Discrete(x), spaces.Discrete(y)))

        super().__init__(observation_space=observation_space, default_reward=default_reward, seed=seed,
                         initial_state=initial_state, obstacles=obstacles, finals=finals, action_space=action_space)

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

                    # Set a position
                    state = (x, y)

                    if state == self.current_state:
                        icon = self.icons['CURRENT']
                    elif state in self.obstacles:
                        icon = self.icons['BLOCK']
                    elif state in self.finals.keys():
                        icon = self.finals[state]
                    else:
                        icon = self.icons['BLANK']

                    # Show col
                    print('| {} '.format(icon), end='')

                # New row
                print('|')

            # End render
            print('')

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next position with current position and action given.
        Default is 4-neighbors (UP, LEFT, DOWN, RIGHT)
        :param state: If a position is given, do action from that position.
        :param action: from action_space
        :return: a new position (or old if is invalid action)
        """

        # Unpack position
        position = state if state else self.current_state

        next_position, is_valid = self.next_position(action=action, position=position)

        if not self.observation_space.contains(next_position) or not is_valid:
            next_position = position

        # Return (x, y) position
        return next_position

    def next_position(self, action: int, position: tuple) -> (tuple, bool):
        """
        Given a action an initial position, returns the next position and if the next position is valid or not.
        :param action:
        :param position:
        :return:
        """

        # Unpack position
        x, y = position

        # Do movement
        if action == self.actions['UP']:
            y -= 1
        elif action == self.actions['RIGHT']:
            x += 1
        elif action == self.actions['DOWN']:
            y += 1
        elif action == self.actions['LEFT']:
            x -= 1

        # Pack next position
        next_position = x, y

        # If exists obstacles, then next_position must be in self.obstacles
        is_obstacle = bool(self.obstacles) and next_position in self.obstacles

        # If is an obstacle, then isn't valid.
        return next_position, not is_obstacle

    def states(self) -> set:
        """
        Return all possible states of this environment.
        :return:
        """

        # Unpack spaces
        x_position, y_position = self.observation_space.spaces

        # Return all spaces
        return {(x, y) for x in range(x_position.n) for y in range(y_position.n)} - (
            self.obstacles.union(set(self.finals.keys())))

    def reachable_states(self, state: tuple, action: int) -> set:
        return {self.next_state(action=action, state=state)}
