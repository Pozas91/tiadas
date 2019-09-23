"""The environment is a grid of 10 rows and 11 columns. The agent controls a 
submarine searching for undersea treasures. There are multiple treasure locations
with varying treasure values. There are two objectives - to minimise the number 
of steps taken to reach the treasure, and to maximise the value of the treasure. 
Each episode starts with the vessel in the top left state, and ends when a 
treasure location is reached or after a given number of actions. Four actions 
are available to the agent - moving one square to the left, right, up or down. 
Any action that would cause the agent to leave the grid will leave its position 
unchanged. The reward received by the agent is a 2-element vector. The first 
element is a time penalty, which adds -1 on each step. The second element is 
the treasure value  of the state the agent moves into (zero for any non-terminal
state).

Notice that states are represented by a tuple (a, b), where b stands for the
row (y-axis), and a for the column (x-axis).                                                     

Episodes end whenever a final state (with a treasure) is reached.

Refernce: 
    Empirical Evaluation methods for multi-objective reinforcement learning algorithms
    (Vamplew, Dazeley, Berry, Issabekov and Dekker) 2011
"""

from models import Vector
from .env_mesh import EnvMesh


class DeepSeaTreasure(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Pareto optimal policy vector-values
    pareto_optimal = [
        Vector([-1, 1]), Vector([-3, 2]), Vector([-5, 3]), Vector([-7, 5]), 
        Vector([-8, 8]), Vector([-9, 16]), Vector([-13, 24]), Vector([-14, 50]), 
        Vector([-17, 74]), Vector([-19, 124])
    ]

    def __init__(self, initial_state: tuple = (0, 0), 
                 default_reward: tuple = (0,), 
                 seed: int = 0,
                 steps_limit: int = 1000):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        """

        # Dictionary with final states as keys, and treasure amounts as values.
        finals = {
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

        obstacles = frozenset()
        obstacles = obstacles.union([(0, y) for y in range(2, 11)])
        obstacles = obstacles.union([(1, y) for y in range(3, 11)])
        obstacles = obstacles.union([(2, y) for y in range(4, 11)])
        obstacles = obstacles.union([(3, y) for y in range(5, 11)])
        obstacles = obstacles.union([(4, y) for y in range(5, 11)])
        obstacles = obstacles.union([(5, y) for y in range(5, 11)])
        obstacles = obstacles.union([(6, y) for y in range(8, 11)])
        obstacles = obstacles.union([(7, y) for y in range(8, 11)])
        obstacles = obstacles.union([(8, y) for y in range(10, 11)])

        mesh_shape = (10, 11)

        # Default reward plus time (time_inverted, treasure_value)
        default_reward = (-1,) + default_reward
        default_reward = Vector(default_reward)

        super().__init__(mesh_shape=mesh_shape,  
                         seed=seed, 
                         initial_state=initial_state, 
                         default_reward=default_reward,
                         finals=finals, 
                         obstacles=obstacles)

        # Step counter and limit
        self.steps_limit = steps_limit
        self.steps = 0

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (time_inverted, treasure_value), final, info)
        """

        # Increment step counter
        self.steps += 1

        # Initialize rewards as vector
        rewards = self.default_reward.copy()

        # Get new state
        new_state = self.next_state(action=action)

        # Update current state
        self.current_state = new_state

        # Get treasure value
        rewards[1] = self.finals.get(self.current_state, self.default_reward[1])

        # Set info
        info = {}

        # Check is_final
        final = self.is_final(self.current_state)

        return self.current_state, rewards, final, info

    def reset(self) -> tuple:
        """
        Reset environment to initial state.
        :return:
        """
        self.steps = 0
        self.current_state = self.initial_state
        return self.current_state

    def is_final(self, state: tuple = None) -> bool:
        """
        Return True if the state given is final or the maximum number of
        steps has been reached, False otherwise.
        :param state:
        :return:
        """
        return state in self.finals.keys() or self.steps >= self.steps_limit
