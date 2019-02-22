from .env_mesh import EnvMesh


class BuridanAss(EnvMesh):
    _actions = {
        'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'STAY': 4
    }

    def __init__(self, mesh_shape=(3, 3), initial_state=(1, 1), default_reward=0., seed=0, p_stolen=.9,
                 n_appear=10, stolen_penalty=-.5, walking_penalty=-1., hunger_penalty=-1., last_ate_limit=9):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        :param p_stolen: Probability to stole food if not are visible.
        :param n_appear: Number of time-steps until food is regenerated.
        :param stolen_penalty: Penalty when the food are stolen.
        :param walking_penalty: Penalty for each step.
        :param hunger_penalty: Penalty for not eat.
        """

        finals = {
            # state: There are? or Time-steps to regenerate
            (0, 0): True,
            (2, 2): True
        }

        super().__init__(mesh_shape, seed, default_reward=default_reward, initial_state=initial_state, finals=finals)

        self.p_stolen = p_stolen
        self.n_appear = n_appear
        self.walking_penalty = walking_penalty
        self.stolen_penalty = stolen_penalty
        self.hunger_penalty = hunger_penalty
        self.last_ate_limit = last_ate_limit

        # Last time that donkey ate.
        self.last_ate = 0

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
        new_state = self._next_state(action=action)
        complex_state[0] = new_state

        # Get all available food
        available_food = [
            state_with_food for state_with_food in self.finals.keys() if
            isinstance(self.finals.get(state_with_food), bool)
        ]

        # If donkey stay in state with food, eat it.
        if self.current_state in available_food and self._actions.get('STAY') == action:
            # Set last_ate to 0
            self.last_ate = 0

            # Update data
            self.finals.update({
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
        rewards[2] = 0. if self._actions.get('STAY') == action else -1.

        # Set last ate
        complex_state[2] = self.last_ate

        # Have they stolen the food?
        rewards[1] = self.__stolen_food()

        # Set info
        info = {}

        # If there isn't more food left, it is a final state
        final = any(not isinstance(self.finals.get(state), bool) for state in self.finals.keys())

        return tuple(complex_state), rewards, final, info
        # return new_state, rewards, final, info

    def reset(self):
        self.current_state = self.initial_state
        self.last_ate = 0

        for state_with_food in self.finals.keys():
            self.finals.update({state_with_food: True})

        return self.current_state

    def __near_food(self) -> list:
        """
        Return near states where there are food.
        :return:
        """
        state = self.current_state

        return [
            state_with_food for state_with_food in self.finals.keys() if
            self.__are_8_neighbours(state_a=state_with_food, state_b=state) and isinstance(
                self.finals.get(state_with_food), bool)
        ]

    def __stolen_food(self) -> float:
        """
        Modified food dictionary and return penalize by stolen food
        :return: penalize
        """

        penalize = 0.

        visible_food = self.__near_food()

        # Get all food that exists and aren't in donkey's vision.
        unprotected_food = [state_with_food for state_with_food in self.finals.keys() if
                            isinstance(self.finals.get(state_with_food), bool) and state_with_food not in visible_food]

        for food_state in unprotected_food:
            # Get a random uniform number [0., 1.]

            if self.np_random.uniform() >= self.p_stolen:
                self.finals.update({food_state: self.n_appear})
                penalize += self.stolen_penalty

        return penalize

    def __regenerate_food(self):
        """
        Regenerate food if is necessary
        :return:
        """
        # Check all food states
        for state_with_food in self.finals.keys():
            # Get data from food dictionary
            data = self.finals.get(state_with_food)

            # If not there are food, and is time to regenerate, regenerate it.
            if not isinstance(data, bool) and data <= 0:
                data = True

            # Decrement time-step to appear new food
            elif not isinstance(data, bool) and data > 0:
                data -= 1

            self.finals.update({state_with_food: data})

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
