import operator

import numpy as np


class Agent:
    __icons = {
        'BLANK': ' ', 'BLOCK': '■', 'FINAL': '$', 'CURRENT': '☺', 'UP': '↑', 'RIGHT': '→', 'DOWN': '↓', 'LEFT': '←',
        'STAY': '×'
    }

    def __init__(self, environment, alpha=0.1, epsilon=0.1, gamma=0.6, seed=0, default_action=0, default_reward=0.,
                 states_to_observe=None, max_iterations=None):

        # Check alpha
        assert 0.0 < alpha <= 1.0
        self.alpha = alpha

        self.epsilon = epsilon
        self.gamma = gamma
        self.environment = environment
        self.default_action = default_action
        self.default_reward = default_reward

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
        self.generator = np.random.RandomState(seed=seed)

        # Initialize to Q-Learning Dictionary
        self.q = dict()

        # Rewards history data
        self.rewards_history = list()

    def __setstate__(self, state) -> None:
        """
        Set an initial state
        :param state:
        :return:
        """
        self.state = state

    def select_action(self) -> int:
        """
        Select best action with a little e-greedy policy.
        :return:
        """

        if self.generator.uniform(low=0., high=1.) < self.epsilon:
            # Get random action to explore possibilities
            action = self.environment.action_space.sample()

        else:
            # Get best action to exploit reward.
            action = self._get_best_action()

        return action

    def episode(self) -> None:
        """
        Run an episode complete until get a final step
        :return:
        """

        # Reset mesh
        self.state = self.environment.reset()

        # Condition to stop episode
        is_final_state = False

        # Reset iterations
        self._reset_iterations()

        while not is_final_state:

            # Increment iterations
            self.iterations += 1

            # Get an action
            action = self.select_action()

            # Do step on environment
            next_state, reward, is_final_state, info = self.environment.step(action=action)

            # Transform reward
            reward = self._processing_reward(reward=reward)

            # Append to rewards history
            self.rewards_history.append(reward)

            # Get old value
            old_value = self.q.get(self.state, {}).get(action, self.default_reward)

            # Get next max value
            next_max = self._get_best_value(state=next_state)

            # Calc new value apply Q-Learning formula:
            # Q(St, At) <- (1 - alpha) * Q(St, At) + alpha * (r + y * Q(St_1, action))
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

            # Prepare new data
            new_data = {action: new_value}

            # If we know this state
            if self.state in self.q:
                # Update value for the action done.
                self.q.get(self.state).update(new_data)
            else:
                # Set a new dictionary for this state
                self.q.update({
                    self.state: new_data
                })

            # Update state
            self.state = next_state

            # Check timeout
            if self.max_iterations is not None and not is_final_state:
                is_final_state = self.iterations >= self.max_iterations

        # Append new data
        for state, data in self.states_to_observe.items():
            # Add to data Best value (V max)
            data.append(self._get_best_value(state))

            # Update dictionary
            self.states_to_observe.update({
                state: data
            })

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

                if hasattr(self.environment, 'obstacles') and state in self.environment.obstacles:
                    icon = self.__icons.get('BLOCK')
                elif state not in self.q.keys():
                    icon = '-'
                else:
                    # Get best action
                    icon = self._get_best_action(state=state)

                # Show col
                print('| {} '.format(icon), end='')

            # New row
            print('|')

        # New line
        print('')

    def show_crude_policy(self):
        """
        Show all states with it's best action
        :return:
        """
        # For each state in q
        for state in self.q.keys():
            best_action = self._get_best_action(state=state)

            print("State: {} -> Action: {}".format(state, best_action))

    def reset(self):
        """
        Reset agent, forgetting previous q-values
        :return:
        """
        self.rewards_history = list()
        self.q = dict()
        self.state = self.environment.reset()
        self.iterations = 0

    def _get_best_action(self, state=None) -> int:
        """
        Return best action for q and state given.
        :return:
        """

        if state is None:
            state = self.state

        # Get best action.
        data = self.q.get(state, {})

        if data:
            # Get max by value, and get it's action
            action = max(self.q.get(state).items(), key=operator.itemgetter(1))[0]
        else:
            # If don't know best action, get a random action
            # action = self.environment.action_space.sample()
            action = self.default_action

        return action

    def _get_best_value(self, state) -> float:
        """
        Return best value for q and state given
        :return:
        """

        # Get data
        data = self.q.get(state, {})

        if data:
            # Get max by value, and get it
            value = max(self.q.get(state).items(), key=operator.itemgetter(1))[1]
        else:
            value = self.default_reward

        return value

    def get_v(self) -> float:
        """
        Get V(0, 0)
        :return:
        """
        initial_state = self.environment.initial_state
        return self._get_best_value(state=initial_state)

    def _reset_iterations(self):
        self.iterations = 0

    def _processing_reward(self, reward):
        """
        Processing reward function.
        :param reward:
        :return:
        """
        return reward
