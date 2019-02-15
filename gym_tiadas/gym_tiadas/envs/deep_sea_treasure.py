import gym

from gym import spaces
from gym.utils import seeding


class DeepSeaTreasure(gym.Env):
    __actions = {
        'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3
    }
    __icons = {'BLANK': ' ', 'BLOCK': '■', 'TREASURE': '$', 'CURRENT': '☺'}

    def __init__(self, initial_observation=(0, 0), default_reward=0., seed=0, time_limit=1000):
        """
        :param initial_observation:
        :param default_reward:
        :param seed:
        """

        self.action_space = spaces.Discrete(len(self.__actions))

        # Mesh of 10 cols and 11 rows
        self.observation_space = spaces.Tuple((
            spaces.Discrete(10), spaces.Discrete(11)
        ))

        self.default_reward = default_reward

        assert isinstance(initial_observation, tuple) and self.observation_space.contains(initial_observation)
        self.initial_state = initial_observation
        self.current_state = self.initial_state

        # List of all treasures and its reward.
        self.treasures = {
            (0, 1): 1,
            (1, 2): 2,
            (2, 3): 3,
            (3, 4): 5,
            (4, 4): 8,
            (5, 4): 16,
            (6, 7): 24,
            (7, 7): 50,
            (8, 9): 74,
            (9, 10): 124,
        }

        self.obstacles = frozenset()
        self.obstacles = self.obstacles.union([(0, y) for y in range(2, 11)])
        self.obstacles = self.obstacles.union([(1, y) for y in range(3, 11)])
        self.obstacles = self.obstacles.union([(2, y) for y in range(4, 11)])
        self.obstacles = self.obstacles.union([(3, y) for y in range(5, 11)])
        self.obstacles = self.obstacles.union([(4, y) for y in range(5, 11)])
        self.obstacles = self.obstacles.union([(5, y) for y in range(5, 11)])
        self.obstacles = self.obstacles.union([(6, y) for y in range(8, 11)])
        self.obstacles = self.obstacles.union([(7, y) for y in range(8, 11)])
        self.obstacles = self.obstacles.union([(8, y) for y in range(10, 11)])

        # Time inverted in find a treasure
        self.time = 0
        self.time_limit = time_limit

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

    def step(self, action) -> (object, [float, float], bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (time_inverted, treasure_value), final, info)
        """

        # (time_inverted, treasure_value)
        rewards = [0., 0.]

        # Get new state
        new_state = self.__next_state(action=action)

        # Update previous state
        self.current_state = new_state
        self.time += 1

        # Get time inverted
        rewards[0] = -self.time

        # Get treasure value
        rewards[1] = self.treasures.get(self.current_state, self.default_reward)

        # Set info
        info = {}

        # If agent is in treasure or time limit has reached
        final = self.current_state in self.treasures.keys() or self.time >= self.time_limit

        return self.current_state, rewards, final, info

    def reset(self):
        self.current_state = self.initial_state
        self.time = 0

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
                elif state in self.obstacles:
                    icon = self.__icons.get('BLOCK')
                elif state in self.treasures.keys():
                    icon = self.treasures.get(state)
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

        if not self.observation_space.contains(new_state) or new_state in self.obstacles:
            # New state is invalid.
            new_state = self.current_state

        # Return (x, y) position
        return new_state
