import operator

import numpy as np


class AgentMultiObjective:
    __icons = {
        'BLANK': ' ', 'BLOCK': '■', 'FINAL': '$', 'CURRENT': '☺', 'UP': '↑', 'RIGHT': '→', 'DOWN': '↓', 'LEFT': '←',
    }

    __actions = {
        'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3
    }

    def __init__(self, environment, alpha=0.1, epsilon=0.1, gamma=0.6, seed=0, default_action=0,
                 states_to_observe=None, rewards_weights=None):

        # Check alpha
        assert 0.0 < alpha <= 1.0
        self.alpha = alpha

        self.epsilon = epsilon
        self.gamma = gamma
        self.environment = environment
        self.default_action = default_action

        # Create dictionary of states to observe
        if states_to_observe is None:
            self.states_to_observe = dict()
        else:
            self.states_to_observe = {state: list() for state in states_to_observe}

        # Set weights to rewards
        self.rewards_weights = rewards_weights

        # Current Agent State if the initial state of environment
        self.state = self.environment.reset()

        # Initialize Random Generator with `seed` as initial seed.
        self.generator = np.random.RandomState(seed=seed)

        # Initialize to Q-Learning Dictionary
        self.q = dict()

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
            action = self.__get_best_action()

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

        while not is_final_state:
            # Get an action
            action = self.select_action()

            # Do step on environment
            next_state, rewards, is_final_state, info = self.environment.step(action=action)

            # If not weights define, all rewards have same weight
            weights = [1] * len(rewards) if self.rewards_weights is None else self.rewards_weights

            # Apply weights to rewards to get only one reward
            reward = np.sum(np.multiply(rewards, weights))

            # Get old value
            old_value = self.q.get(self.state, {}).get(action, self.environment.default_reward)

            # Get next max value
            next_max = self.__get_best_value(state=next_state)

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

        # Append new data
        for state, data in self.states_to_observe.items():
            # Add to data Best value (V max)
            data.append(self.__get_best_value(state))

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

                if state in self.environment.obstacles:
                    icon = self.__icons.get('BLOCK')

                else:
                    # Get best action
                    best_action = self.__get_best_action(state=state)

                    if best_action == self.__actions.get('UP'):
                        icon = self.__icons.get('UP')
                    elif best_action == self.__actions.get('RIGHT'):
                        icon = self.__icons.get('RIGHT')
                    elif best_action == self.__actions.get('DOWN'):
                        icon = self.__icons.get('DOWN')
                    else:
                        icon = self.__icons.get('LEFT')

                # Show col
                print('| {} '.format(icon), end='')

            # New row
            print('|')

        # New line
        print('')

    def show_crude_policy(self):

        # For each state in q
        for state in self.q.keys():
            best_action = self.__get_best_action(state=state)

            if best_action == self.__actions.get('UP'):
                icon = self.__icons.get('UP')
            elif best_action == self.__actions.get('RIGHT'):
                icon = self.__icons.get('RIGHT')
            elif best_action == self.__actions.get('DOWN'):
                icon = self.__icons.get('DOWN')
            elif best_action == self.__actions.get('LEFT'):
                icon = self.__icons.get('LEFT')
            else:
                icon = self.__icons.get('STAY')

            print("State: {} -> Action: {}".format(state, icon))

    def __get_best_action(self, state=None) -> int:
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
            action = self.environment.action_space.sample()

        return action

    def __get_best_value(self, state) -> float:
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
            value = self.environment.default_reward

        return value
