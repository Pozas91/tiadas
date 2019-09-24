"""The donkey is in the center square of the 3 x 3 grid. There are food piles on the diagonally opposite squares. The
food is visible only from the neighboring squares in the eight directions. If the donkey moves away from the
neighboring square of a food pile, there is a certain probability `p_stolen` with which the food is stolen. Food is
regenerated once every `n_appear` time-steps. The donkey has to strike a compromise between minimizing the three
different costs: hunger, lost food, and walking. A state is a tuple (s, f, t), where s stands for the square in which
the donkey is present, f for food in the two piles, and t for the time since the donkey last ate food. If t = 9,
it is not incremented and the donkey incurs a penalty of -1 per time step till it eats the food when t is reset to 0.
The actions are move up, down, left, right, and stay. It is assumed that if the donkey chooses to stay at a square
with food, then it eats the food. `p_stolen` is set to 0.9, `n_appear` is set to 10. The stolen penalty is -0.5 per
plate and walking penalty is -1 per step.

FINAL STATE: No food to eat.

REF: Dynamic Preferences in Multi-Criteria Reinforcement Learning (Sriraam Natarajan)"""
from models import VectorDecimal
from .env_mesh import EnvMesh


class BuridanAss(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'STAY': 4}

    def __init__(self, initial_state: tuple = (1, 1), default_reward: tuple = (0., 0., 0.), seed: int = 0,
                 p_stolen: float = .9, n_appear: int = 10, stolen_penalty: float = -.5, walking_penalty: float = -1,
                 hunger_penalty: float = -1, last_ate_limit: int = 9):
        """
        :param initial_state:
        :param default_reward: (hunger, stolen, walking)
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

        mesh_shape = (3, 3)
        default_reward = VectorDecimal(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, default_reward=default_reward, initial_state=initial_state,
                         finals=finals)

        self.p_stolen = p_stolen
        self.n_appear = n_appear
        self.walking_penalty = walking_penalty
        self.stolen_penalty = stolen_penalty
        self.hunger_penalty = hunger_penalty
        self.last_ate_limit = last_ate_limit

        # Last time that donkey ate.
        self.last_ate = 0

    def step(self, action: int) -> (tuple, VectorDecimal, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Initialize rewards as vector
        rewards = self.default_reward.copy()

        # (state, states_visible_with_food, last_ate)
        complex_state = [None, None, None]

        # Get new state
        new_state = self.next_state(action=action)
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
        rewards[2] = self.default_reward[2] if self._actions.get('STAY') == action else self.walking_penalty

        # Set last ate
        complex_state[2] = self.last_ate

        # Have they stolen the food?
        rewards[1] = self.__stolen_food()

        # Set info
        info = {}

        # Check is_final
        final = self.is_final()

        return tuple(complex_state), rewards, final, info

    def reset(self) -> tuple:
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

    def __regenerate_food(self) -> None:
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
    def __are_8_neighbours(state_a: tuple, state_b: tuple) -> bool:
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

    def is_final(self, state: tuple = None) -> bool:
        """
        If there is not more food left, it is a final state
        :param state:
        :return:
        """
        return all(not isinstance(self.finals.get(state), bool) for state in self.finals.keys())
