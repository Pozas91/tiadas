"""
Episodic version of ResourceGathering.

REF: 'A temporal difference method for multi-objective reinforcement learning'
    (Manuela Ruíz-Montiel, Lawrence Mandow, José-Luis Pérez-de-la-Cruz 2017)

FINAL STATES: Return at home with any object or attacked by enemy.
"""
import itertools

import gym

import spaces
from models import Vector
from . import ResourceGathering


class ResourceGatheringEpisodic(ResourceGathering):

    def __init__(self, initial_state: tuple = ((2, 4), (0, 0), False), default_reward: tuple = (0, 0, 0), seed: int = 0,
                 p_attack: float = 0.1, steps_limit: int = 1000, mesh_shape: tuple = (5, 5)):
        """
        :param initial_state: Initial state where start the agent.
        :param default_reward: (enemy_attack, gold, gems)
        :param seed: Seed used for np.random.RandomState method.
        :param p_attack: Probability that a enemy attacks when agent stay in an enemy position.
        """

        # Build the observation space (position(x, y), quantity(gold, gems), attacked)
        observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Tuple(
                    (gym.spaces.Discrete(mesh_shape[0]), gym.spaces.Discrete(mesh_shape[1]))
                ),
                gym.spaces.Tuple(
                    (gym.spaces.Discrete(2), gym.spaces.Discrete(2))
                ),
                spaces.Boolean()
            )
        )

        super().__init__(initial_state=initial_state, default_reward=default_reward, seed=seed, p_attack=p_attack,
                         observation_space=observation_space)

        # Define final states
        self.finals = self.checkpoints_states.copy().union({((2, 4), (0, 0), True)})

        self.steps = 0
        self.steps_limit = steps_limit

    def _checkpoints_states(self) -> set:
        """
        Return states where the agent will get favorable reward.
        :return:
        """
        return set(itertools.product({self.home_position}, {(1, 0), (0, 1), (1, 1)}, {False}))

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Increment steps
        self.steps += 1

        # Initialize reward as vector
        reward = self.default_reward.copy()

        # Update previous position
        self.current_state = self.next_state(action=action)

        # Check if is final state
        final = self.is_final()

        # Final by attack
        if self.current_state[2]:
            reward[0] = -1

        # Final by timeout
        elif self.steps >= self.steps_limit:
            reward[:] = 0
            final = True

        # Another final
        elif final or self.current_state in self.checkpoints_states:
            reward[1:3] = self.current_state[1]
            final = True

        # Set extra
        info = {}

        return self.current_state, reward, final, info

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next position with current position and action given. Default is 4-neighbors (UP, LEFT, DOWN, RIGHT)
        :param state: If a position is given, do action from that position.
        :param action: from action_space
        :return: a new position (or old if is invalid action)
        """

        # Unpack complex state (position, objects(gold, gem), attacked)
        position, objects, attacked = state if state else self.current_state

        # Calc next position
        next_position, is_valid = self.next_position(action=action, position=position)

        # If the next_position isn't valid, reset to the previous position
        if not self.observation_space[0].contains(next_position) or not is_valid:
            next_position = position

        if next_position in self.gold_positions:
            objects = 1, objects[1]

        elif next_position in self.gem_positions:
            objects = objects[0], 1

        elif next_position in self.enemies_positions and self.p_attack >= self.np_random.uniform():
            next_position, objects, attacked = self.home_position, (0, 0), True

        return next_position, objects, attacked

    def reset(self) -> tuple:
        """
        Reset environment to zero.
        :return:
        """

        # Super method call
        super().reset()

        # Reset steps
        self.steps = 0

        return self.current_state

    def transition_reward(self, state: tuple, action: int, next_state: tuple) -> Vector:
        """
        Return reward for reach `next_state` from `state` using `action`.

        :param state: initial position
        :param action: action to do
        :param next_state: next position reached
        :return:
        """
        # Initialize reward as vector
        reward = self.default_reward.copy()
        # Unpack next state
        _, _, attacked = next_state

        # Check if has been attacked
        if attacked:
            reward[:] = -1, 0, 0

        # Check if next state is on checkpoints states
        elif next_state in self.checkpoints_states:
            reward[1:3] = next_state[1]

        # Final timeout
        elif self.steps >= self.steps_limit:
            reward[1:3] = 0

        return reward

    def states(self) -> set:
        """
        Return all states from this environment
        :return:
        """

        # Unpack spaces
        x_position, y_position = self.observation_space[0]

        # Calc basic states
        basic_states = set(
            (x, y) for x in range(x_position.n) for y in range(y_position.n)
        ).difference(self.obstacles)

        # Calc product of basic states with objects
        states = set(
            itertools.product(basic_states, {(0, 0), (0, 1), (1, 0), (1, 1)}, {False})
        ).difference(
            self.finals
        ).difference(
            set(
                # Cannot be in gold positions without gold.
                itertools.product(self.gold_positions, {(0, 0), (0, 1)}, {False})
            ).union(
                # Cannot be in gem positions without gem.
                itertools.product(self.gem_positions, {(0, 0), (1, 0)}, {False})
            ).union(
                # Cannot be in home position with gem and/or gold.
                self.checkpoints_states
            )
        )

        # Return all spaces
        return states

    def is_final(self, state: tuple = None) -> bool:
        """
        Return True if position given is terminal
        :return:
        """
        state = state if state else self.current_state
        return state in self.finals

    def reachable_states(self, state: tuple, action: int) -> set:
        """
        Return all reachable states for pair (state, action) given.
        :param state:
        :param action:
        :return:
        """

        reachable_states = set()

        if (state[0] == (3, 1) or state[0] == (3, 0)) and action == self.actions['UP']:
            reachable_states.add(((3, 0), state[1], False))
            reachable_states.add((self.home_position, (0, 0), True))
        elif state[0] == (3, 1) and action == self.actions['LEFT']:
            reachable_states.add(((2, 1), state[1], False))
            reachable_states.add((self.home_position, (0, 0), True))
        elif state[0] == (4, 0) and action == self.actions['LEFT']:
            reachable_states.add(((3, 0), state[1], False))
            reachable_states.add((self.home_position, (0, 0), True))
        elif state[0] == (2, 2) and action == self.actions['UP']:
            reachable_states.add(((2, 1), state[1], False))
            reachable_states.add((self.home_position, (0, 0), True))
        elif state[0] == (1, 1) and action == self.actions['RIGHT']:
            reachable_states.add(((2, 1), state[1], False))
            reachable_states.add((self.home_position, (0, 0), True))
        elif state[0] == (2, 0) and action == self.actions['DOWN']:
            reachable_states.add(((2, 1), state[1], False))
            reachable_states.add((self.home_position, (0, 0), True))
        elif state[0] == (2, 0) and action == self.actions['RIGHT']:
            reachable_states.add(((3, 0), state[1], False))
            reachable_states.add((self.home_position, (0, 0), True))
        else:
            reachable_states.add(self.next_state(action=action, state=state))

        # Return all possible states reachable with any action
        return reachable_states
