"""The donkey is in the center square of the 3 x 3 grid. There are food piles on the diagonally opposite squares. The
food is visible only from the neighboring squares in the eight directions. If the donkey moves away from the
neighboring square of a food pile, there is a certain probability `p_stolen` with which the food is stolen. Food is
regenerated once every `n_appear` time-steps. The donkey has to strike a compromise between minimizing the three
different costs: hunger, lost food, and walking. A position is a tuple (s, f, t), where s stands for the square in which
the donkey is present, f for food in the two piles, and t for the time since the donkey last ate food. If t = 9,
it is not incremented and the donkey incurs a penalty of -1 per time step till it eats the food when t is reset to 0.
The actions are move up, down, left, right, and stay. It is assumed that if the donkey chooses to stay at a square
with food, then it eats the food. `p_stolen` is set to 0.9, `n_appear` is set to 10. The stolen penalty is -0.5 per
plate and walking penalty is -1 per step.

FINAL STATE: No food to eat.

REF: Dynamic Preferences in Multi-Criteria Reinforcement Learning (Sriraam Natarajan)"""

import gym

import spaces
from models import VectorDecimal
from .env_mesh import EnvMesh


class BuridanAss(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3, 'STAY': 4}

    def __init__(self, initial_state: tuple = ((1, 1), {(0, 0), (2, 2)}, 0), default_reward: tuple = (0., 0., 0.),
                 p_stolen: float = .9, n_appear: int = 10, stolen_penalty: float = -.5, walking_penalty: float = -1,
                 hunger_penalty: float = -1., last_ate_limit: int = 9, seed: int = 0):
        """
        :param default_reward: (hunger, stolen, walking)
        :param p_stolen: Probability to stole food if not are visible.
        :param n_appear: Number of time-steps until food is regenerated.
        :param stolen_penalty: Penalty when the food are stolen.
        :param walking_penalty: Penalty for each step.
        :param hunger_penalty: Penalty for not eat.
        """

        mesh_shape = (3, 3)
        default_reward = VectorDecimal(default_reward)

        finals = set()

        for x in range(mesh_shape[0]):
            for y in range(mesh_shape[1]):
                for last_ate in range(last_ate_limit + 1):
                    finals.add((
                        (x, y), frozenset(), last_ate
                    ))

        # Build the observation space (position(x, y), visible food (bag), last ate (discrete))
        observation_space = gym.spaces.Tuple(
            (
                gym.spaces.Tuple(
                    (gym.spaces.Discrete(mesh_shape[0]), gym.spaces.Discrete(mesh_shape[1]))
                ),
                spaces.Bag([
                    frozenset(), frozenset({(0, 0)}), frozenset({(2, 2)}), frozenset({(0, 0), (2, 2)})
                ]),
                gym.spaces.Discrete(last_ate_limit + 1)
            )
        )

        super().__init__(mesh_shape=mesh_shape, default_reward=default_reward, finals=finals,
                         observation_space=observation_space, initial_state=initial_state, seed=seed)

        self.p_stolen = p_stolen
        self.n_appear = n_appear
        self.walking_penalty = walking_penalty
        self.stolen_penalty = stolen_penalty
        self.hunger_penalty = hunger_penalty

        self.food_counter = {
            (0, 0): 0,
            (2, 2): 0
        }

    def next_state(self, action: int, state: tuple = None) -> tuple:

        # Unpack complex position
        position, _, last_ate = state if state else self.current_state

        # Default value is same position
        next_position = position

        # Get all positions with food
        positions_with_food = self.__positions_with_food()

        # If donkey is stay on position with food, eat it.
        if action == self.actions['STAY'] and position in positions_with_food:
            last_ate = 0

            # Update food train_data
            self.food_counter.update({
                # Set time-steps to appear again (plus 1 to check later)
                position: self.n_appear + 1
            })

            # Positions with food are equals that before positions without this position
            positions_with_food = positions_with_food - {position}

        # Otherwise, is more hungry
        else:

            # Increment hungry
            last_ate = min(last_ate + 1, self.observation_space[2].n - 1)

            # If donkey moves calc next position
            if action != self.actions['STAY']:

                # Calc next position
                next_position, is_valid = self.next_position(action=action, position=position)

                # If the next_position isn't valid, reset to the previous position
                if not self.observation_space[0].contains(next_position) or not is_valid:
                    next_position = position

        # Is necessary regenerate food?
        self.__regenerate_food()

        return next_position, positions_with_food, last_ate

    def __positions_with_food(self) -> frozenset:
        return frozenset(filter(lambda key: self.food_counter[key] <= 0, self.food_counter.keys()))

    def step(self, action: int) -> (tuple, VectorDecimal, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Initialize reward as vector
        reward = self.default_reward.copy()

        # Have they stolen the food?
        reward[1] = self.__stolen_food()

        # Get new position
        next_position, positions_with_food, last_ate = self.next_state(action=action)

        # If donkey is hungry, has penalty
        if last_ate >= (self.observation_space[2].n - 1):
            reward[0] += self.hunger_penalty

        # Pack position
        self.current_state = next_position, positions_with_food, last_ate

        # If action is different to stay, donkey is walking and have a penalize.
        reward[2] = self.default_reward[2] if self.actions['STAY'] == action else self.walking_penalty

        # Set extra
        info = {}

        # Check is_final
        final = self.is_final()

        return self.current_state, reward, final, info

    def reset(self) -> tuple:
        # Reset to initial seed
        self.seed(seed=self.initial_seed)

        for state_with_food in self.food_counter.keys():
            self.food_counter.update({state_with_food: 0})

        self.current_state = self.initial_state
        return self.current_state

    def __near_food(self, position: tuple) -> frozenset:
        """
        Return near states where there are food.
        :return:
        """

        return frozenset(
            filter(
                lambda position_with_food: self.__are_8_neighbours(position_with_food, position) and self.food_counter[
                    position_with_food] <= 0, self.food_counter.keys()
            )
        )

    def __stolen_food(self) -> float:
        """
        Modified food dictionary and return penalize by stolen food
        :return: penalize
        """

        penalize = 0.

        visible_food = self.__near_food(position=self.current_state[0])

        # Get all food that exists and aren't in donkey's vision.
        unprotected_food = frozenset(
            filter(lambda position_with_food: self.food_counter[position_with_food] <= 0, self.food_counter.keys())
        ) - visible_food

        for food_state in unprotected_food:
            # Get a random uniform number [0., 1.]

            if self.np_random.uniform() >= self.p_stolen:
                self.food_counter.update({food_state: self.n_appear})
                penalize += self.stolen_penalty

        return penalize

    def __regenerate_food(self) -> None:
        """
        Regenerate food if is necessary
        :return:
        """
        # Check all food positions
        for position_with_food in self.food_counter.keys():

            # Get train_data from food dictionary
            data = self.food_counter[position_with_food]

            if data > 0:
                data -= 1

            self.food_counter.update({position_with_food: data})

    @staticmethod
    def __are_8_neighbours(position_a: tuple, position_b: tuple) -> bool:
        """
        Check if position_a and position_b are neighbours
        :param position_a:
        :param position_b:
        :return:
        """

        # Decompose position a
        x_a, y_a = position_a

        # Generate 8-neighbours
        a_neighbours = [(x, y) for x in range(x_a - 1, x_a + 2) for y in range(y_a - 1, y_a + 2)]

        # Check if b is neighbour to a
        return position_b in a_neighbours

    def states(self) -> set:

        # States
        states = list()

        # Unpack position from rest of spaces
        x_position, y_position = self.observation_space[0]

        # Positions with food
        positions_with_food = self.observation_space[1].items

        # Calc positions
        positions = {(x, y) for x in range(x_position.n) for y in range(y_position.n)}

        for position in positions:
            for position_with_food in positions_with_food:
                for last_ate in range(self.observation_space[2].n):

                    invalid_state = position == (0, 0) and position_with_food in ({(0, 0), (2, 2)}, {(2, 2)})
                    invalid_state |= position == (1, 0) and position_with_food in ({(0, 0), (2, 2)}, {(2, 2)})
                    invalid_state |= position == (0, 1) and position_with_food in ({(0, 0), (2, 2)}, {(2, 2)})
                    invalid_state |= position == (2, 2) and position_with_food in ({(0, 0), (2, 2)}, {(0, 0)})
                    invalid_state |= position == (2, 1) and position_with_food in ({(0, 0), (2, 2)}, {(0, 0)})
                    invalid_state |= position == (1, 2) and position_with_food in ({(0, 0), (2, 2)}, {(0, 0)})
                    invalid_state |= position == (0, 2) and position_with_food in ({(0, 0), (2, 2)}, {(0, 0)}, {(2, 2)})
                    invalid_state |= position == (2, 0) and position_with_food in ({(0, 0), (2, 2)}, {(0, 0)}, {(2, 2)})
                    invalid_state |= position == (2, 0) and position_with_food in ({(0, 0), (2, 2)}, {(0, 0)}, {(2, 2)})
                    invalid_state |= position in {(1, 0), (2, 1), (1, 2), (0, 1)} and last_ate < 1
                    invalid_state |= position in {(2, 0), (0, 2)} and last_ate < 2

                    if not invalid_state:
                        full_state = position, position_with_food, last_ate
                        states.append(full_state)

        # Return all spaces
        return set(states) - self.finals
