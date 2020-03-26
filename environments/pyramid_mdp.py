"""
The Pyramid MDP is a new and simple multi-objective benchmark, which we introduce
in this paper. The agent starts in the down-left position, denoted by a black dot at (0; 0), and it can choose any of
the four cardinal directions (up, down, left and right). The transition function is stochastic
so that with a probability of 0:95 the selected action is performed and with a probability
of 0:05 a random transition is executed to a neighboring state. The red dots represent
terminal states. The reward scheme is bi-objective and returns a reward drawn from a
Normal distribution with u = -1 and o = 0.01 for both objectives, unless a terminal state
is reached. In that case, the x and y position of the terminal state is returned for the
first and second objective, respectively.

REF: Multi-objective reinforcement learning using sets of pareto dominating policies (Kristof Van Moffaert,
Ann Nowé) 2014

HV REFERENCE: (-20, -20)
"""
import gym

import utils.environments as ue
from environments import EnvMesh
from models import Vector


class PyramidMDP(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Pareto optimal policy vector-values
    pareto_optimal = []

    # Experiments common hypervolume reference
    hv_reference = Vector((-20, -20))

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (-1, -1), seed: int = 0,
                 n_transition: float = 0.95, diagonals: int = 9, action_space: gym.spaces = None):
        """
        :param initial_state: Initial state where start the agent.
        :param default_reward: (objective 1, objective 2)
        :param seed: Seed used for np.random.RandomState method.
        :param n_transition: if is 1, always do the action indicated. (Original is about 0.6)
        :param diagonals: Number of diagonals to be used to build this environment (allows experimenting with an
                        identical environment, but considering only the first k diagonals) (By default 9 - all).
        """

        # the original full-size environment.
        mesh_shape = (min(max(diagonals + 1, 1), 10), min(max(diagonals + 1, 1), 10))

        # Dictionary with final states as keys, and treasure amounts as values.
        diagonals_states = {x for x in zip(range(0, diagonals + 1, 1), range(diagonals, -1, -1))}

        # Generate finals states with its reward
        finals = {state: (Vector(state) + 1) * 10 for state in diagonals_states}

        # Pareto optimal
        PyramidMDP.pareto_optimal = {Vector(state) + 1 for state in diagonals_states}

        # Filter obstacles states
        obstacles = frozenset(
            (x, y) for x, y in finals.keys() for y in range(y, diagonals + 1) if (x, y) not in finals
        )

        # Default reward (objective_1, objective_2)
        default_reward = Vector(default_reward)

        # Transaction
        assert 0 <= n_transition <= 1.
        self.n_transition = n_transition

        super().__init__(mesh_shape=mesh_shape, initial_state=initial_state, default_reward=default_reward,
                         finals=finals, obstacles=obstacles, seed=seed, action_space=action_space)

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (position, (time_inverted, treasure_value), final, extra)
        """

        # Get probability action
        action = self.__probability_action(action=action)

        # Initialize rewards as vector
        reward = self.default_reward.copy()

        # Update current position
        self.current_state = self.next_state(action=action)

        # Get treasure value
        reward = self.finals.get(self.current_state, reward)

        # Set extra
        info = {}

        # Check is_final
        final = self.is_final(self.current_state)

        return self.current_state, reward, final, info

    def __probability_action(self, action: int) -> int:
        """
        Decide probability action after apply probabilistic p_stochastic.
        :param action:
        :return:
        """

        # Get a random uniform number [0., 1.]
        random = self.np_random.uniform()

        # If random is greater than self.n_transition, get a random action
        if random > self.n_transition:
            action = self.action_space.sample()

        return action

    def transition_reward(self, state: tuple, action: int, next_state: tuple) -> Vector:
        """
        Return reward for reach `next_state` from `position` using `action`.

        :param state: initial position
        :param action: action to do
        :param next_state: next position reached
        :return:
        """
        # Default reward
        return self.finals.get(next_state, self.default_reward.copy())

    def transition_probability(self, state: tuple, action: int, next_state: tuple) -> float:
        """
        Return probability to reach `next_state` from `position` using `action`.
        :param state: initial position
        :param action: action to do
        :param next_state: next position reached
        :return:
        """
        # Probability
        desired_probability = self.n_transition

        desired_transition = (
                (action == self.actions['UP'] and ue.is_on_up_or_same_position(
                    state=state, next_state=next_state
                )) or
                (action == self.actions['RIGHT'] and ue.is_on_right_or_same_position(
                    state=state, next_position=next_state
                )) or
                (action == self.actions['DOWN'] and ue.is_on_down_or_same_position(
                    state=state, next_state=next_state
                )) or
                (action == self.actions['LEFT'] and ue.is_on_left_or_same_position(
                    state=state, next_state=next_state
                ))
        )

        if not desired_transition:
            desired_probability = (1. - self.n_transition) / self.action_space.n

        return desired_probability

    def reachable_states(self, state: tuple, action: int) -> set:
        """
        Return all reachable states for pair (state, a) given.
        :param state:
        :param action:
        :return:
        """
        # Set current state with state indicated
        self.current_state = state

        # Get all actions available
        actions = self.action_space.copy()

        # Return all possible states reachable with any action
        return {self.next_state(action=a, state=state) for a in actions}
