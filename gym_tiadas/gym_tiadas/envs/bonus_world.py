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
from .env_mesh import EnvMesh


class BonusWorld(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    def __init__(self, mesh_shape=(9, 9), initial_state=(0, 0), default_reward=(0., 0.), seed=0):
        """
        :param initial_state:
        :param default_reward:
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

        obstacles = set()
        obstacles.add((2, 2))
        obstacles.add((2, 3))
        obstacles.add((3, 2))

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

        super().__init__(mesh_shape, seed, initial_state=initial_state, default_reward=default_reward, finals=finals,
                         obstacles=obstacles)

    def step(self, action) -> (object, [float, float, float], bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (time_inverted, treasure_value), final, info)
        """

        # (objective 1, objective 2, time)
        rewards = [0, 0, 0]

        # Get new state
        new_state = self._next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Check if the agent has activated the bonus
        if self.current_state in self.bonus:
            self.bonus_activated = True

        # If agent is in pit, it's returned at initial state.
        if self.current_state in self.pits:
            self.current_state = self.initial_state

        # Get time inverted
        rewards[2] = -1

        # Get treasure value
        rewards[0], rewards[1] = self.finals.get(self.current_state, self.default_reward)

        # If the bonus is activated, double the reward.
        if self.bonus_activated:
            rewards[0] *= 2
            rewards[1] *= 2

        # Set info
        info = {}

        # If agent is in treasure
        final = self.current_state in self.finals.keys()

        return self.current_state, rewards, final, info

    def reset(self):
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        self.bonus_activated = False

        return self.current_state
