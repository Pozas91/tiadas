from scipy.spatial import distance

from .env_mesh import EnvMesh


class MoPuddleWorld(EnvMesh):
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    def __init__(self, mesh_shape=(20, 20), finish_reward=10., penalize_non_goal=-1., seed=0, final_state=(19, 0)):

        obstacles = frozenset()
        obstacles = obstacles.union([(x, y) for x in range(0, 11) for y in range(3, 7)])
        obstacles = obstacles.union([(x, y) for x in range(6, 10) for y in range(2, 14)])

        super().__init__(mesh_shape, seed, obstacles=obstacles)

        self.penalize_non_goal = penalize_non_goal
        self.final_reward = finish_reward

        self.final_state = final_state
        self.current_state = self.reset()

    def step(self, action) -> (object, [float, float], bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (non_goal_reached, puddle_penalize), final, info)
        """

        # (non_goal_reached, puddle_penalize)
        rewards = [0., 0.]

        # Get new state
        new_state = self._next_state(action=action)

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
