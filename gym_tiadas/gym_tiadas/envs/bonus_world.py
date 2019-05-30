"""Like DST it is a 2D episodic grid environment, but it has three objectives rather than two. Each episode starts
with the agent in the location marked ’S’. The agent can move in the four cardinal directions, and receives a reward
of −1 for the time objective on every time-step. When reaching a terminal state the agent receives the rewards
specified in that cell for the other two objectives. In addition the rewards in the terminal states are doubled in
magnitude if the agent has activated the bonus by visiting the cell marked ’X2’. The black cells near the bonus
indicate walls which the agent cannot pass through. Similarly the agent cannot leave the bounds of the grid. Finally
the cells marked ’PIT’ indicate pits – if the agent enters one of these cells the bonus is deactivated, and the agent
returns to the start state. A tabular representation of this environment has 162 discrete states – 81 for the cells
of the grid when the agent has not activated the bonus, and 81 for the same cells when the bonus has been activated.
The set of Pareto-optimal policies and the corresponding thresholds are listed in Table 2. It can be seen that
trade-offs exist between all three objectives. Note that not all optimal policies require the agent to activate the
bonus.

FINAL STATE: To reach a final state.

REF: Vamplew et al (2017b)"""
from models import Vector
from .env_mesh import EnvMesh


class BonusWorld(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0, 0), seed: int = 0):
        """
        :param initial_state:
        :param default_reward: (objective 1, objective 2)
        :param seed:
        """

        # List of all treasures and its reward.
        finals = {
            (8, 0): (1, 9),
            (8, 2): (3, 9),
            (8, 4): (5, 9),
            (8, 6): (7, 9),
            (8, 8): (9, 9),

            (0, 8): (9, 1),
            (2, 8): (9, 3),
            (4, 8): (9, 5),
            (6, 8): (9, 7),
        }

        # Define mesh shape
        mesh_shape = (9, 9)

        # Set obstacles
        obstacles = frozenset([(2, 2), (2, 3), (3, 2)])

        # Default reward plus time (objective 1, objective 2, time)
        default_reward += (-1,)
        default_reward = Vector(default_reward)

        super().__init__(mesh_shape=mesh_shape, seed=seed, default_reward=default_reward, initial_state=initial_state,
                         finals=finals, obstacles=obstacles)

        # Pits marks which returns the agent to the start location.
        self.pits = [
            (7, 1), (7, 3), (7, 5), (1, 7), (3, 7), (5, 7)
        ]

        # X2 bonus
        self.bonus = [
            (3, 3)
        ]

        # Bonus is activated?
        self.bonus_activated = False

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (time_inverted, treasure_value), final, info)
        """

        # Initialize rewards as vector
        rewards = self.default_reward.copy()

        # Get new state
        new_state = self.next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Check if the agent has activated the bonus
        if self.current_state in self.bonus:
            self.bonus_activated = True

        # If agent is in pit, it's returned at initial state.
        if self.current_state in self.pits:
            self.current_state = self.initial_state

        # Get treasure value
        rewards[0], rewards[1] = self.finals.get(self.current_state, (self.default_reward[0], self.default_reward[1]))

        # If the bonus is activated, double the reward.
        if self.bonus_activated:
            rewards[0] *= 2
            rewards[1] *= 2

        # Set info
        info = {}

        # Check is_final
        final = self.is_final(self.current_state)

        return self.current_state, rewards, final, info

    def reset(self) -> tuple:
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        self.bonus_activated = False

        return self.current_state

    def is_final(self, state: tuple = None) -> bool:
        """
        Is final if agent is on final state.
        :param state:
        :return:
        """
        return state in self.finals.keys()

    def get_dict_model(self) -> dict:
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment data
        del data['pits']
        del data['bonus']

        return data
