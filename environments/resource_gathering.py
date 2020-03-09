"""
An agent begins at the home location in a 2D grid, and can move one square at a time in each of the four cardinal
directions. The agent'state task is to collect either or both of two resources (gold and gems) which are available at fixed
locations, and return home with these resources. The environment contains two locations at which an enemy attack may
occur, with a 10% probability. If an attack happens, the agent loses any resources currently being carried and is
returned to the home location. The reward vector is ordered as [enemy, gold, gems] and there are four possible rewards
which may be received on entering the home location.

• [−1, 0, 0] in case of an enemy attack;
• [0, 1, 0] for returning home with gold but no gems;
• [0, 0, 1] for returning home with gems but no gold;
• [0, 1, 1] for returning home with both gold and gems.

FINAL STATE: Doesn't have final state. Continuous task.

REF: Empirical Evaluation methods for multi-objective reinforcement learning algorithms
    (Vamplew, Dazeley, Berry, Issabekov and Dekker) 2011
"""
import itertools

import gym

from models import Vector
from .env_mesh import EnvMesh


class ResourceGathering(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Reference
    hv_reference = Vector((-10, -10, -10))

    def __init__(self, initial_state: tuple = ((2, 4), (0, 0)), default_reward: tuple = (0, 0, 0), seed: int = 0,
                 p_attack: float = 0.1, mesh_shape: tuple = (5, 5), gold_positions: frozenset = frozenset({(2, 0)}),
                 gem_positions: frozenset = frozenset({(4, 1)})):
        """
        :param initial_state:
        :param default_reward: (enemy_attack, gold, gems)
        :param seed:
        :param p_attack: Probability that a enemy attacks when agent stay in an enemy position.
        """

        default_reward = Vector(default_reward)

        # Build the observation space (position(x, y), quantity(gold, gems))
        observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Tuple(
                    (gym.spaces.Discrete(mesh_shape[0]), gym.spaces.Discrete(mesh_shape[1]))
                ),
                gym.spaces.Tuple(
                    (gym.spaces.Discrete(2), gym.spaces.Discrete(2))
                )
            )
        )

        # Define final states
        finals = frozenset()

        # Super constructor call.
        super().__init__(mesh_shape=mesh_shape, seed=seed, initial_state=initial_state, default_reward=default_reward,
                         observation_space=observation_space, finals=finals)

        # Positions where there are gold.
        self.gold_positions = gold_positions

        # Positions where there is a gem.
        self.gem_positions = gem_positions

        # States where there are enemies_positions
        self.enemies_positions = {(3, 0), (2, 1)}
        self.p_attack = p_attack
        self.home_position = (2, 4)
        self.attacked = False

        self.checkpoints_states = set(itertools.product({self.home_position}, {(1, 0), (0, 1), (1, 1)}))

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Set attacked to False
        self.attacked = False

        # Initialize reward as vector
        reward = self.default_reward.copy()

        # Update previous position
        self.current_state = self.next_state(action=action)

        if self.attacked:
            reward[0] = -1

        # Check if is final state
        final = self.is_final()

        # Check is_final
        if final or self.current_state in self.checkpoints_states:
            reward[1], reward[2] = self.current_state[1]

        # Set extra
        info = {}

        return self.current_state, reward, final, info

    def next_state(self, action: int, state: tuple = None) -> tuple:

        # Unpack complex state (position, objects(gold, gem))
        position, objects = state if state else self.current_state

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
            next_position, objects = self.initial_state
            next_position = self.home_position
            self.attacked = True

        return next_position, objects

    def reset(self) -> tuple:
        """
        Reset environment to zero.
        :return:
        """

        # Reset to initial seed
        self.seed(seed=self.initial_seed)

        self.current_state = self.initial_state
        self.attacked = False

        return self.current_state

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment train_data
        del data['gold_positions']
        del data['gem_positions']
        del data['enemies_positions']
        del data['home_position']
        del data['checkpoints_states']

        return data

    def states(self) -> set:

        # Unpack spaces
        x_position, y_position = self.observation_space[0]

        # Calc basic states
        basic_states = set(
            (x, y) for x in range(x_position.n) for y in range(y_position.n)
        ).difference(self.obstacles)

        # Calc product of basic states with objects
        states = set(
            itertools.product(basic_states, {(0, 0), (0, 1), (1, 0), (1, 1)})
        ).difference(
            self.finals
        ).difference(
            set(
                # Cannot be in gold positions without gold.
                itertools.product(self.gold_positions, {(0, 0), (0, 1)})
            ).union(
                # Cannot be in gem positions without gem.
                itertools.product(self.gem_positions, {(0, 0), (1, 0)})
            ).union(
                # Cannot be in home position with gem and/or gold.
                self.checkpoints_states
            )
        )

        # Return all spaces
        return states

    def warning_action(self, state: tuple, action: int):
        return ((state[0] == (3, 1) or state[0] == (3, 0)) and action == self.actions['UP']) or \
               (state[0] == (3, 1) and action == self.actions['LEFT']) or \
               (state[0] == (4, 0) and action == self.actions['LEFT']) or \
               (state[0] == (2, 2) and action == self.actions['UP']) or \
               (state[0] == (1, 1) and action == self.actions['RIGHT']) or \
               (state[0] == (2, 0) and action == self.actions['DOWN']) or \
               (state[0] == (2, 0) and action == self.actions['RIGHT'])

    def transition_reward(self, state: tuple, action: int, next_state: tuple) -> Vector:
        # Initialize reward as vector
        reward = self.default_reward.copy()

        # Default attacked is false
        self.attacked = False

        if self.warning_action(state=state, action=action) and next_state[0] == self.home_position:
            self.attacked = True

        if self.attacked:
            reward[0], reward[1], reward[2] = -1, 0, 0
        elif next_state in self.checkpoints_states:
            reward[1], reward[2] = next_state[1]

        return reward

    def transition_probability(self, state: tuple, action: int, next_state: tuple) -> float:

        transition_probability = 1.

        if self.warning_action(state=state, action=action):
            transition_probability = self.p_attack if (next_state[0] == self.home_position) else 1. - self.p_attack

        return transition_probability

    def reachable_states(self, state: tuple, action: int) -> set:

        reachable_states = set()

        if (state[0] == (3, 1) or state[0] == (3, 0)) and action == self.actions['UP']:
            reachable_states.add(((3, 0), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (3, 1) and action == self.actions['LEFT']:
            reachable_states.add(((2, 1), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (4, 0) and action == self.actions['LEFT']:
            reachable_states.add(((3, 0), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (2, 2) and action == self.actions['UP']:
            reachable_states.add(((2, 1), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (1, 1) and action == self.actions['RIGHT']:
            reachable_states.add(((2, 1), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (2, 0) and action == self.actions['DOWN']:
            reachable_states.add(((2, 1), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (2, 0) and action == self.actions['RIGHT']:
            reachable_states.add(((3, 0), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        else:
            reachable_states.add(self.next_state(action=action, state=state))

        # Return all possible states reachable with any action
        return reachable_states

    def is_final(self, state: tuple = None) -> bool:
        """
        Return always false (No episodic task)
        :return:
        """
        state = state if state else self.current_state
        return state in self.finals
