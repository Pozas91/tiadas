"""
Base model to define environments which observation_space as mesh.
Mesh is a grid of columns per rows, where the minimum value is (0, 0), and maximum value is (columns - 1, rows - 1).
Top-left corner is (0, 0), top-right corner is (columns - 1, 0), bottom-left corner is (0, rows - 1) and
bottom-right corner is (columns - 1, rows - 1).

Action space is a discrete number. Allow numbers from [0, n).

Finals is a dictionary which structure as follows:
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
        :param mesh_shape: A tuple where first component is a x-axis, and second component is y-axis.
        :param default_reward: Default reward that return environment when a reward is not defined.
        :param seed: Initial seed.
        :param initial_state: First state where agent start.
        :param obstacles: States where agent can not to be.
        :param finals: States where agent finish an epoch.
        """

        # Create the mesh
        x, y = mesh_shape
        observation_space = gym.spaces.Tuple((spaces.Discrete(x), spaces.Discrete(y)))

        super().__init__(observation_space=observation_space, default_reward=default_reward, seed=seed,
                         initial_state=initial_state, obstacles=obstacles, finals=finals)

    def render(self, mode: str = 'human') -> None:
        """
        Render environment
        :param mode:
        :return:
        """

        if mode == 'human':
            # Get cols (x) and rows (y) from observation space
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
        Calc next state with current state and action given. Default is 4-neighbors (UP, LEFT, DOWN, RIGHT)
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
