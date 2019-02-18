import gym
import numpy as np
from gym import spaces
from gym.utils import seeding


class ResourceGathering(gym.Env):
    __actions = {
        'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3
    }

    __icons = {'BLANK': ' ', 'BLOCK': '■', 'TREASURE': '$', 'CURRENT': '☺', 'ENEMY': '×', 'HOME': 'µ'}

    __treasures = {'GOLD': 0, 'GEM': 1}

    def __init__(self, initial_state=(2, 4), default_reward=0., seed=0, enemies=None, golds=None, gems=None,
                 p_attack=0.1):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        """

        if enemies is None:
            enemies = [(3, 0), (2, 1)]

        self.enemies = enemies
        self.p_attack = p_attack

        self.action_space = spaces.Discrete(len(self.__actions))

        # 5x5 Grid
        self.observation_space = spaces.Tuple((
            spaces.Discrete(5), spaces.Discrete(5)
        ))

        self.default_reward = default_reward

        assert isinstance(initial_state, tuple) and self.observation_space.contains(initial_state)
        self.home_state = initial_state
        self.current_state = self.home_state

        if golds is None:
            golds = {(2, 0): True}

        self.golds = golds

        if gems is None:
            gems = {(4, 1): True}

        self.gems = gems

        # [enemy_attack, gold, gems]
        self.state = [0., 0., 0.]

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

        # Final
        final = False

        # Calc rewards
        rewards = np.multiply(self.state, 0.)

        # Get new state
        new_state = self.__next_state(action=action)

        # Update previous state
        self.current_state = new_state

        if self.current_state in self.enemies:
            final = self.__enemy_attack()
            rewards = np.multiply(self.state, 1.)
        elif self.current_state in self.golds.keys():
            self.__get_gold()
        elif self.current_state in self.gems.keys():
            self.__get_gem()
        elif self.__at_home():
            final = self.__is_checkpoint()
            rewards = np.multiply(self.state, 1.)

        # Set info
        info = {}

        return (self.current_state, tuple(self.state)), rewards, final, info

    def reset(self):
        self.current_state = self.home_state
        self.state = np.multiply(self.state, 0.)
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
                elif state in self.golds.keys():
                    icon = self.__icons.get('TREASURE')
                elif state in self.gems.keys():
                    icon = self.__icons.get('TREASURE')
                elif state in self.enemies:
                    icon = self.__icons.get('ENEMY')
                elif state == self.home_state:
                    icon = self.__icons.get('HOME')
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

    def __enemy_attack(self) -> bool:
        """
        Check if enemy attack you
        :return:
        """

        final = False

        if self.np_random.uniform() > self.p_attack:
            self.reset()
            self.state[0] = -1.
            final = True

        return final

    def __get_gold(self):
        """
        Check if agent can take the gold.
        :param state:
        :return:
        """

        # Check if there is a gold
        if self.golds.get(self.current_state, False):
            self.state[1] += 1.
            self.golds.update({self.current_state: False})

    def __get_gem(self):
        """
        Check if agent can take the gem.
        :return:
        """

        # Check if there is a gem
        if self.gems.get(self.current_state, False):
            self.state[2] += 1.
            self.gems.update({self.current_state: False})

    def __at_home(self) -> bool:
        """
        Check if agent is at home
        :return:
        """

        return self.current_state == self.home_state

    def __is_checkpoint(self) -> bool:
        """
        Check if is final state (has gold, gem or both)
        :return:
        """

        return (self.state[1] >= 0. or self.state[2] >= 0.) and self.__at_home()
