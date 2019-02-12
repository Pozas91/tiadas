import numpy as np


class AgentDiscrete:
    __icons = {
        'BLANK': ' ', 'BLOCK': '■', 'FINAL': '$', 'CURRENT': '☺', 'UP': '↑', 'RIGHT': '→', 'DOWN': '↓', 'LEFT': '←'
    }

    __actions = {
        'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3
    }

    def __init__(self, environment, alpha=0.1, epsilon=0.1, gamma=0.6, seed=0):

        # Check alpha
        assert 0.0 < alpha <= 1.0
        self.alpha = alpha

        self.epsilon = epsilon
        self.gamma = gamma
        self.environment = environment

        # Current Agent State if the initial state of environment
        self.state = self.environment.reset()

        # Initialize Random Generator with `seed` as initial seed.
        self.generator = np.random.RandomState(seed=seed)

        # Initialize to zero Q-Learning Table
        self.q_table = np.zeros([self.environment.observation_space.n, self.environment.action_space.n])
        self.q_table += self.environment.default_reward

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
            action = np.argmax(self.q_table[self.state])

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
            next_state, reward, is_final_state, info = self.environment.step(action=action)

            # Get old value: Q(St, At)
            old_value = self.q_table[self.state, action]

            # Get next max value: Q(St_1, At)
            next_max = np.max(self.q_table[next_state])

            # Calc new value apply Q-Learning formula:
            # Q(St, At) <- (1 - alpha) * Q(St, At) + alpha * (r + y * Q(St_1, action))
            new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)

            # Update data
            self.q_table[self.state, action] = new_value

            # Update state
            self.state = next_state

    def show_q(self) -> None:
        """
        Show Q-Data
        :return:
        """
        print(self.q_table)

    def show_policy(self) -> None:
        """
        Show policy's matrix
        :return:
        """

        # Get rows and cols from states
        cols, rows = self.environment.shape

        for y in range(rows):
            for x in range(cols):

                state = self.__tuple_to_discrete(t=(x, y))

                if state in self.environment.obstacles:
                    icon = self.__icons.get('BLOCK')
                elif state in self.environment.finals.keys():
                    icon = self.__icons.get('FINAL')
                else:

                    # Get best action
                    best_action = np.argmax(self.q_table[state])

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

    def __tuple_to_discrete(self, t):
        """
        Convert the tuple given to discrete space
        :param t: Tuple (x, y)
        :return:
        """
        return t[1] * self.environment.shape[0] + t[0]
