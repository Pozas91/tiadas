import gym

from gym import spaces
from gym.utils import seeding
from scipy.spatial import distance


class MoPuddleWorld(gym.Env):
    __actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}
    __icons = {'BLANK': ' ', 'BLOCK': '■', 'REWARD': '$', 'CURRENT': '☺'}

    def __init__(self, finish_reward=10., penalize_non_goal=-1., seed=0, final_state=(19, 0)):

        self.penalize_non_goal = penalize_non_goal
        self.final_reward = finish_reward
        self.action_space = spaces.Discrete(len(self.__actions))

        # Mesh of 10 cols and 11 rows
        self.observation_space = spaces.Tuple((
            spaces.Discrete(20), spaces.Discrete(20)
        ))

        self.obstacles = frozenset()
        self.obstacles = self.obstacles.union([(x, y) for x in range(0, 11) for y in range(3, 7)])
        self.obstacles = self.obstacles.union([(x, y) for x in range(6, 10) for y in range(2, 14)])

        self.final_state = final_state
        self.current_state = self.reset()

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
        :return: (state, (non_goal_reached, puddle_penalize), final, info)
        """

        # (non_goal_reached, puddle_penalize)
        rewards = [0., 0.]

        # Get new state
        new_state = self.__next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # If agent is in treasure or time limit has reached
        final = self.current_state == self.final_state

        # Set final reward
        rewards[0] = self.final_reward if final else self.penalize_non_goal

        if self.current_state in self.obstacles:
            x_space, y_space = self.observation_space.spaces
            # Get all spaces
            all_space = [(x, y) for x in range(x_space.n) for y in range(y_space.n)]
            # Get free spaces
            free_spaces = list(set(all_space) - self.obstacles)
            # Start with infinite distance
            min_distance = float('inf')

            # For each free space
            for state in free_spaces:
                min_distance = min(min_distance, distance.cityblock(self.current_state, state))

            # Set penalization per distance
            rewards[1] = -min_distance

            # Set info
        info = {}

        return self.current_state, rewards, final, info

    def reset(self):
        """
        Get random non-goal state to current_value
        :return:
        """
        random_space = self.observation_space.sample()

        while random_space == self.final_state:
            random_space = self.observation_space.sample()

        self.current_state = random_space
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
