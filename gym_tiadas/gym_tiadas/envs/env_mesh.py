import gym
from gym import spaces
from gym.utils import seeding


class EnvMesh(gym.Env):
    _actions = dict()
    _icons = {'BLANK': ' ', 'BLOCK': '■', 'TREASURE': '$', 'CURRENT': '☺', 'ENEMY': '×', 'HOME': 'µ', 'FINAL': '$'}

    def __init__(self, mesh_shape: tuple, seed=None, initial_state=None, obstacles=None, finals=None, default_reward=0.,
                 default_action=0):

        # Set action space
        self.action_space = spaces.Discrete(len(self._actions))

        # Create the mesh
        x, y = mesh_shape
        self.observation_space = spaces.Tuple((spaces.Discrete(x), spaces.Discrete(y)))

        # Prepare random seed
        self.np_random = None
        self.seed(seed=seed)

        # Set current environment state
        assert initial_state is None or self.observation_space.contains(initial_state)
        self.initial_state = initial_state
        self.current_state = self.initial_state

        # Set finals states
        assert finals is None or all([self.observation_space.contains(final) for final in finals.keys()])
        self.finals = finals

        # Set obstacles
        assert obstacles is None or all([self.observation_space.contains(obstacle) for obstacle in obstacles])
        self.obstacles = obstacles

        # Defaults
        self.default_action = default_action
        self.default_reward = default_reward

        # Reset environment
        self.reset()

    def step(self, action):
        raise NotImplemented

    def seed(self, seed=None):
        """
        Generate seed
        :param seed:
        :return:
        """
        self.np_random, seed = seeding.np_random(seed=seed)
        return [seed]

    def reset(self):
        """
        Reset environment
        :return:
        """
        raise NotImplemented

    def render(self, mode='human'):
        """
        Render environment
        :param mode:
        :return:
        """

        if mode == 'human':
            # Get cols (x) and rows (y) from observation space
            cols, rows = self.observation_space.spaces[0].n, self.observation_space.spaces[1].n

            for y in range(rows):
                for x in range(cols):

                    # Set a state
                    state = (x, y)

                    if state == self.current_state:
                        icon = self._icons.get('CURRENT')
                    elif state in self.obstacles:
                        icon = self._icons.get('BLOCK')
                    elif state in self.finals.keys():
                        icon = self.finals.get(state)
                    else:
                        icon = self._icons.get('BLANK')

                    # Show col
                    print('| {} '.format(icon), end='')

                # New row
                print('|')

            # End render
            print('')

    def _next_state(self, action) -> object:
        """
        Calc next state with current state and action given. Default is 4-neighbors (UP, LEFT, DOWN, RIGHT)
        :param action: from action_space
        :return: a new state (or old if is invalid action)
        """

        # Get my position
        x, y = self.current_state

        # Do movement
        if action == self._actions.get('UP'):
            y -= 1
        elif action == self._actions.get('RIGHT'):
            x += 1
        elif action == self._actions.get('DOWN'):
            y += 1
        elif action == self._actions.get('LEFT'):
            x -= 1

        # Set new state
        new_state = x, y

        # If exists obstacles, then new_state must be in self.obstacles (p => q)
        is_obstacle = not hasattr(self, 'obstacles') or new_state in self.obstacles

        if not self.observation_space.contains(new_state) and not is_obstacle:
            # New state is invalid.
            new_state = self.current_state

        # Return (x, y) position
        return new_state
