"""
Variant of Mo Puddle World for Acyclic agents.
"""
from scipy.spatial import distance

from models import VectorFloat
from spaces import DynamicSpace
from .env_mesh import EnvMesh


class MoPuddleWorldAcyclic(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1}

    def __init__(self, default_reward: tuple = (10, 0), penalize_non_goal: float = -1, seed: int = 0,
                 final_state: tuple = (19, 0)):
        """
        :param default_reward: (non_goal_reached, puddle_penalize)
        :param penalize_non_goal: While agent does not reach a final state get a penalize.
        :param seed:
        :param final_state: This environment only has a final state.
        """

        self.final_state = final_state
        mesh_shape = (20, 20)
        default_reward = VectorFloat(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, default_reward=default_reward)

        self.puddles = frozenset()
        self.puddles = self.puddles.union([(x, y) for x in range(0, 11) for y in range(3, 7)])
        self.puddles = self.puddles.union([(x, y) for x in range(6, 10) for y in range(2, 14)])
        self.penalize_non_goal = penalize_non_goal

        self.current_state = self.reset()

        # Trying improve performance
        self.dynamic_action_space = DynamicSpace([])
        self.dynamic_action_space.seed(seed=seed)

        # Unpack spaces
        x_space, y_space = self.observation_space.spaces
        # Get all spaces
        all_space = [(x, y) for x in range(x_space.n) for y in range(y_space.n)]
        # Get free spaces
        self.free_spaces = list(set(all_space) - self.puddles)

    def step(self, action: int) -> (tuple, VectorFloat, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (non_goal_reached, puddle_penalize), final, info)
        """

        # Initialize rewards as vector
        rewards = self.default_reward.copy()

        # Get new state
        new_state = self.next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # If agent is in treasure
        final = self.is_final(self.current_state)

        # Set final reward
        if not final:
            rewards[0] = self.penalize_non_goal

        # if the current state is in an puddle
        if self.current_state in self.puddles:

            # Min distance found!
            min_distance = min(distance.cityblock(self.current_state, state) for state in self.free_spaces)

            # Set penalization per distance
            rewards[1] = -min_distance

        # Set info
        info = {}

        return self.current_state, rewards, final, info

    def reset(self) -> tuple:
        """
        Get random non-goal state to current_value
        :return:
        """

        while True:
            random_space = self.observation_space.sample()

            if random_space != self.final_state:
                break

        self.current_state = random_space
        return self.current_state

    def is_final(self, state: tuple = None) -> bool:
        """
        Is final if agent is on final state
        :param state:
        :return:
        """
        return state == self.final_state

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next state with current state and action given. Default is 2-neighbors (UP, RIGHT)
        :param state: If a state is given, do action from that state.
        :param action: from action_space
        :return: a new state (or old if is invalid action)
        """

        # Get my position
        x, y = state if state else self.current_state

        # Do movement
        if action == self._actions['RIGHT']:
            x += 1
        elif action == self._actions['UP']:
            y -= 1

        # Set new state
        new_state = x, y

        if not self.observation_space.contains(new_state) or state == new_state:
            raise ValueError("Action/State combination isn't valid.")

        # Return (x, y) position
        return new_state

    @property
    def action_space(self) -> DynamicSpace:
        """
        Get a dynamic action space with only valid actions.
        :return:
        """

        # Get current state
        x, y = self.current_state

        # Setting possible actions
        possible_actions = []

        # Can we go to RIGHT?
        x_right = x + 1

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x_right, y)):
            # We can go to right
            possible_actions.append(self._actions['RIGHT'])

        # Can we go to UP?
        y_up = y - 1

        # Check that y_down is not and obstacle and is into mesh
        if self.observation_space.contains((x, y_up)):
            # We can go to down
            possible_actions.append(self._actions['UP'])

        # Setting to dynamic_space
        self.dynamic_action_space.items = possible_actions

        # Update n length
        self.dynamic_action_space.n = len(possible_actions)

        # Return a list of iterable valid actions
        return self.dynamic_action_space

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment data
        del data['puddles']
        del data['initial_state']
        del data['dynamic_action_space']

        return data
