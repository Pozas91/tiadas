"""
Q-Learning agent to resolve environments trough reinforcement learning.

The data structure of q dictionary is as follows:

{
    state_1: {action_1: reward, action_2: reward, action_3: reward, ...},
    state_2: {action_1: reward, action_2: reward, action_3: reward, ...},
    state_3: {action_1: reward, action_2: reward, action_3: reward, ...},
    ...
}
"""
import matplotlib.pyplot as plt
import numpy as np

from gym_tiadas.gym_tiadas.envs import Environment
from models import Vector


class Agent:
    # Different icons
    __icons = {
        'BLANK': ' ', 'BLOCK': '■', 'FINAL': '$', 'CURRENT': '☺', 'UP': '↑', 'RIGHT': '→', 'DOWN': '↓', 'LEFT': '←',
        'STAY': '×'
    }

    def __init__(self, environment: Environment, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 1.,
                 seed: int = 0, states_to_observe: list = None, max_iterations: int = None):
        """

        :param environment: An environment where agent does any operation.
        :param alpha: Learning rate
        :param epsilon: Epsilon using in e-greedy policy, to explore more states.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param states_to_observe: List of states from that we want to get a graphical output.
        :param max_iterations: Limits of iterations per episode.
        """

        # Learning factor
        assert 0 < alpha <= 1
        # Discount factor
        assert 0 < gamma <= 1
        # Exploration factor
        assert 0 < epsilon <= 1

        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon

        # Set environment
        self.environment = environment

        # To intensive problems
        self.max_iterations = max_iterations
        self.iterations = 0

        # Create dictionary of states to observe
        if states_to_observe is None:
            self.states_to_observe = dict()
        else:
            self.states_to_observe = {state: list() for state in states_to_observe}

        # Current Agent State if the initial state of environment
        self.state = self.environment.reset()

        # Initialize Random Generator with `seed` as initial seed.
        self.seed = seed
        self.generator = np.random.RandomState(seed=seed)

        # Initialize to Q-Learning Dictionary
        self.q = dict()

        # Rewards history data
        self.rewards_history = list()

    def select_action(self, state: object = None) -> int:
        """
        Select best action with a little e-greedy policy.
        :return:
        """

        if self.generator.uniform(low=0., high=1.) < self.epsilon:
            # Get random action to explore possibilities
            action = self.environment.action_space.sample()

        else:
            # Get best action to exploit reward.
            action = self.best_action(state=state)

        return action

    def walk(self) -> list:
        """
        Do a walk follows best current policy
        :return:
        """

        # Reset mesh
        self.state = self.environment.reset()

        # Condition to stop walk
        is_final_state = False

        # Reset iterations
        self.reset_iterations()

        # Rewards history
        history = list()

        while not is_final_state:
            # Increment iterations
            self.iterations += 1

            # Get an action
            action = self.best_action()

            # Do step on environment
            next_state, reward, is_final_state, info = self.environment.step(action=action)

            # Append to rewards history
            history.append(reward)

            # Update state
            self.state = next_state

        return history

    def episode(self) -> None:
        """
        Run an episode complete until get a final step
        :return:
        """

        # Reset environment
        self.state = self.environment.reset()

        # Condition to stop episode
        is_final_state = False

        # Reset iterations
        self.reset_iterations()

        while not is_final_state:

            # Increment iterations
            self.iterations += 1

            # Get an action
            action = self.select_action()

            # Do step on environment
            next_state, reward, is_final_state, info = self.environment.step(action=action)

            # Append to rewards history
            history = self.process_reward(reward=reward)
            self.rewards_history.append(history)

            # Update Q-Dictionary
            self._update_q_dictionary(reward=reward, action=action, next_state=next_state)

            # Update state
            self.state = next_state

            # Check timeout
            if self.max_iterations is not None and not is_final_state:
                is_final_state = self.iterations >= self.max_iterations

        # Append new data
        for state, data in self.states_to_observe.items():
            # Add to data Best value (V max)
            value = self._best_reward(state)

            # Apply function
            value = self.process_reward(value)

            # Add to data Best value (V max)
            data.append(value)

            # Update dictionary
            self.states_to_observe.update({state: data})

    def _update_q_dictionary(self, reward: Vector, action: int, next_state: object) -> None:
        """
        Update Q-Dictionary with new data
        :param reward:
        :param action:
        :param next_state:
        :return:
        """

        # Get old value
        old_value = self.q.get(self.state, {}).get(action, self.environment.default_reward.zero_vector)

        # Get next max value
        next_max = self._best_reward(state=next_state)

        # Calc new value apply Q-Learning formula:
        # Q(St, At) <- (1 - alpha) * Q(St, At) + alpha * (r + y * Q(St_1, action))
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + next_max * self.gamma)

        # Prepare new data
        new_data = {action: new_value}

        # If we know this state
        if self.state in self.q:
            # Update value for the action done.
            self.q.get(self.state).update(new_data)
        else:
            # Set a new dictionary for this state
            self.q.update({self.state: new_data})

    def show_q(self) -> None:
        """
        Show Q-Data
        :return:
        """
        print(self.q)

    def show_policy(self) -> None:
        """
        Show policy's matrix
        :return:
        """

        # Get rows and cols from states
        rows, cols = self.environment.observation_space.spaces[1].n, self.environment.observation_space.spaces[0].n

        for y in range(rows):
            for x in range(cols):

                state = (x, y)

                # Check if our agent has obstacles attribute, if it has, check if state `state` is in an obstacle.
                if hasattr(self.environment, 'obstacles') and state in self.environment.obstacles:
                    icon = self.__icons.get('BLOCK')

                # If state not in Q dictionary, we unknown the state.
                elif state not in self.q.keys():
                    icon = '-'

                # Get best action
                else:
                    icon = self.best_action(state=state)

                # Show col
                print('| {} '.format(icon), end='')

            # New row
            print('|')

        # New line
        print('')

    def show_raw_policy(self) -> None:
        """
        Show all states with it's best action
        :return:
        """
        # For each state in q
        for state in self.q.keys():
            best_action = self.best_action(state=state)

            print("State: {} -> Action: {}".format(state, best_action))

    def reset(self) -> None:
        """
        Reset agent, forgetting previous q-values
        :return:
        """
        self.rewards_history = list()
        self.q = dict()
        self.state = self.environment.reset()
        self.iterations = 0

    def best_action(self, state: object = None) -> int:
        """
        Return best action for q and state given.
        :return:
        """

        # if don't specify a state, get current state.
        if state is None:
            state = self.state

        # Get information about all actions with its rewards.
        possible_actions = self.q.get(state, {})

        # Get unknown actions with default reward
        for action in range(self.environment.action_space.n):
            if action not in possible_actions:
                possible_actions.update({action: self.environment.default_reward.zero_vector})

        # Get max by value, and get it's action
        actions = list()
        max_reward = float('-inf')

        # Check all actions with it's rewards
        for possible_action in possible_actions:

            # Get current Value
            reward = possible_actions.get(possible_action)

            # If current value is close to new value
            if reward.all_close(v2=max_reward):

                # Append another possible action
                actions.append(possible_action)

            # If current value is best than old value
            elif reward > max_reward:

                # Create a new list with current key.
                actions = [possible_action]

            # Update max value
            max_reward = max(max_reward, reward)

        # From best actions get one aleatory.
        action = self.generator.choice(actions)

        return action

    def _best_reward(self, state: object) -> Vector:
        """
        Return best reward for q and state given
        :return:
        """

        # Get information about possible actions
        possible_actions = self.q.get(state, {})

        # Get unknown actions with default reward
        for action in range(self.environment.action_space.n):
            if action not in possible_actions:
                possible_actions.update({action: self.environment.default_reward.zero_vector})

        # Get best action and use it to get best reward.
        action = self.best_action(state=state)
        reward = possible_actions.get(action)

        return reward

    @property
    def v(self) -> Vector:
        """
        Get best value from initial state -> V_max(0, 0)
        :return:
        """
        return self._best_reward(state=self.environment.initial_state)

    def reset_iterations(self) -> None:
        """
        Set iterations to zero.
        :return:
        """
        self.iterations = 0

    def reset_rewards_history(self) -> None:
        """
        Forget rewards history
        :return:
        """
        self.rewards_history = list()

    def process_reward(self, reward: Vector) -> Vector:
        """
        Processing reward function.
        :param reward:
        :return:
        """
        return reward

    def show_observed_states(self) -> None:
        """
        Show graph of observed states
        :return:
        """
        for state, data in self.states_to_observe.items():
            plt.plot(data, label='State: {}'.format(state))

        plt.xlabel('Iterations')
        plt.ylabel('V max')

        plt.legend(loc='upper left')

        plt.show()

    def show_v_values(self) -> None:
        """
        Show Best rewards from Q-Dictionary
        :return:
        """
        for state in self.q.keys():
            # Get rewards
            rewards = self.q.get(state).values()

            # Apply function to each element of list
            # rewards = list(map(self.process_reward, rewards))

            # Print result
            print('State: {} -> V: {}'.format(state, max(rewards)))
