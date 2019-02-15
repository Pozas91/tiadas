import gym

from gym import spaces
from gym.utils import seeding


class BuridanAss(gym.Env):
    __actions = {
        'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'STAY': 4
    }
    __icons = {'BLANK': ' ', 'BLOCK': '■', 'FOOD': '$', 'CURRENT': '☺'}

    def __init__(self, initial_observation=(1, 1), default_reward=0., seed=0, p_stolen=.9, n_appear=10,
                 stolen_penalty=-.5, walking_penalty=-1., hunger_penalty=-1., last_ate_limit=9):
        """
        :param initial_observation:
        :param default_reward:
        :param seed:
        :param p_stolen: Probability to stole food if not are visible.
        :param n_appear: Number of time-steps until food is regenerated.
        :param stolen_penalty: Penalty when the food are stolen.
        :param walking_penalty: Penalty for each step.
        :param hunger_penalty: Penalty for not eat.
        """

        self.action_space = spaces.Discrete(len(self.__actions))
        self.observation_space = spaces.Tuple((
            spaces.Discrete(3), spaces.Discrete(3)
        ))

        self.default_reward = default_reward
        self.p_stolen = p_stolen
        self.n_appear = n_appear
        self.walking_penalty = walking_penalty
        self.stolen_penalty = stolen_penalty
        self.hunger_penalty = hunger_penalty
        self.last_ate_limit = last_ate_limit

        assert isinstance(initial_observation, tuple) and self.observation_space.contains(initial_observation)
        self.initial_state = initial_observation
        self.current_state = self.initial_state

        self.food = {
            # state: There are? or Time-steps to regenerate
            (0, 0): True,
            (2, 2): True
        }

        # Last time that donkey ate.
        self.last_ate = 0

        self.reset()

        self.np_random = None
        self.seed(seed=seed)

    def seed(self, seed=None):
        """
        Generate seed
        :param seed:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed=seed)
        return [seed]

    def step(self, action) -> (object, [float, float, float], bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # (hunger, stolen, walking)
        rewards = [0., 0., 0.]

        # (state, states_visible_with_food, last_ate)
        complex_state = [None, None, None]

        # Get new state
        new_state = self.__next_state(action=action)
        complex_state[0] = new_state

        # Get all available food
        available_food = [
            state_with_food for state_with_food in self.food.keys() if isinstance(self.food.get(state_with_food), bool)
        ]

        # If donkey stay in state with food, eat it.
        if self.current_state in available_food and self.__actions.get('STAY') == action:
            # Set last_ate to 0
            self.last_ate = 0

            # Update data
            self.food.update({
                # Set time-steps to appear again (plus 1 to check later)
                self.current_state: self.n_appear + 1
            })

        # Otherwise, is more hungry
        else:

            # If donkey is hungry, has penalty
            if self.last_ate >= self.last_ate_limit:
                rewards[0] += self.hunger_penalty

            # Otherwise, increment hungry
            else:
                self.last_ate += 1

        # Update previous state
        self.current_state = new_state

        # Is necessary regenerate food?
        self.__regenerate_food()

        # Get states with visible food
        states_with_visible_food = self.__near_food()

        # Set states with visible food
        complex_state[1] = states_with_visible_food[0] if len(states_with_visible_food) == 1 else tuple(
            states_with_visible_food)

        # If action is different to stay, donkey is walking and have a penalize.
        rewards[2] = 0. if self.__actions.get('STAY') == action else -1.

        # Set last ate
        complex_state[2] = self.last_ate

        # Have they stolen the food?
        rewards[1] = self.__stolen_food()

        # Set info
        info = {}

        # If there isn't more food left, it is a final state
        final = all(not isinstance(self.food.get(state), bool) for state in self.food.keys())

        return tuple(complex_state), rewards, final, info
        # return new_state, rewards, final, info

    def reset(self):
        self.current_state = self.initial_state
        self.last_ate = 0

        for state_with_food in self.food.keys():
            self.food.update({state_with_food: True})

        return self.current_state

    def render(self, **kwargs):
        # Get cols (x) and rows (y) from observation space
        cols, rows = self.observation_space.spaces[0].n, self.observation_space.spaces[1].n

        for y in range(rows):
            for x in range(cols):

                # Set a state
                state = (x, y)

                if state == self.current_state:
                    icon = self.__icons.get('CURRENT')
                else:
                    icon = self.__icons.get('BLANK')

                # Show col
                print('| {} '.format(icon), end='')

            # New row
            print('|')

        # End render
        print('')

    def __next_state(self, action) -> (int, int):
        """
        Calc increment or decrement of state, if the new state is out of mesh, or is obstacle, return same state.
        :param action: UP, RIGHT, DOWN, LEFT, STAY
        :return: x, y
        """

        # Get my position
        x, y = self.current_state

        # Do movement
        if action == self.__actions.get('UP'):
            y -= 1
        elif action == self.__actions.get('RIGHT'):
            x += 1
        elif action == self.__actions.get('DOWN'):
            y += 1
        elif action == self.__actions.get('LEFT'):
            x -= 1

        # Set new state
        new_state = x, y

        if not self.observation_space.contains(new_state):
            # New state is invalid.
            new_state = self.current_state

        # Return (x, y) position
        return new_state

    def __near_food(self) -> list:
        """
        Return near states where there are food.
        :return:
        """
        state = self.current_state

        return [
            state_with_food for state_with_food in self.food.keys() if
            self.__are_8_neighbours(state_a=state_with_food, state_b=state) and isinstance(
                self.food.get(state_with_food), bool)
        ]

    def __stolen_food(self) -> float:
        """
        Modified food dictionary and return penalize by stolen food
        :return: penalize
        """

        penalize = 0.

        visible_food = self.__near_food()

        # Get all food that exists and aren't in donkey's vision.
        unprotected_food = [state_with_food for state_with_food in self.food.keys() if
                            isinstance(self.food.get(state_with_food), bool) and state_with_food not in visible_food]

        for food_state in unprotected_food:
            # Get a random uniform number [0., 1.]

            if self.np_random.uniform() >= self.p_stolen:
                self.food.update({food_state: self.n_appear})
                penalize += self.stolen_penalty

        return penalize

    def __regenerate_food(self):
        """
        Regenerate food if is necessary
        :return:
        """
        # Check all food states
        for state_with_food in self.food.keys():
            # Get data from food dictionary
            data = self.food.get(state_with_food)

            # If not there are food, and is time to regenerate, regenerate it.
            if not isinstance(data, bool) and data <= 0:
                data = True

            # Decrement time-step to appear new food
            elif not isinstance(data, bool) and data > 0:
                data -= 1

            self.food.update({state_with_food: data})

    @staticmethod
    def __are_8_neighbours(state_a, state_b) -> bool:
        """
        Check if state_a and state_b are neighbours
        :param state_a:
        :param state_b:
        :return:
        """

        # Decompose state a
        x_a, y_a = state_a

        # Generate 8-neighbours
        a_neighbours = [(x, y) for x in range(x_a - 1, x_a + 2) for y in range(y_a - 1, y_a + 2)]

        # Check if b is neighbour to a
        return state_b in a_neighbours
