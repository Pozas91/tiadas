"""The agent controls a spaceship which starts each episode in the location marked ’S’ and aims to discover a
habitable planet while minimising the amount of radiation to which it is exposed. A penalty of −1 is received for the
radiation objective on all time-steps, except when in a region of high radiation (marked ’R’) when the penalty is
−11. A positive reward is received for the mission success objective whenever a terminal state corresponding to a
planet is reached – the magnitude of this reward reflects the desirability of that planet. If the ship enters a cell
occupied by an asteroid, the ship is destroyed, the episode ends, and the agent receives a mission success reward of
−100. The threshold is applied to the mission success objective, meaning that the agent will attempt to minimise
radiation exposure subject to meeting minimum habitability requirements. in Space Exploration the agent can move to
all eight neighbouring states (i.e. there are eight actions). Also if the agent leaves the bounds of the grid,
it moves to the opposite edge of the grid. For example if the agent moves up from the top row of the grid,
it will move to the bottom row of the same column). """
from .env_mesh import EnvMesh


class SpaceExploration(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    def __init__(self, mesh_shape=(13, 5), initial_state=(5, 2), default_reward=0., seed=0):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        """

        # List of all treasures and its reward.
        finals = {}
        finals.update({(0, i): 20 for i in range(5)})
        finals.update({(9, i): 10 for i in range(3)})
        finals.update({(12, i): 30 for i in range(5)})

        obstacles = {
            (5, 0), (4, 1), (6, 1), (3, 2), (7, 2), (4, 3), (6, 3), (5, 4)
        }

        super().__init__(mesh_shape, seed, initial_state=initial_state, default_reward=default_reward, finals=finals,
                         obstacles=obstacles)

        self.radiations = set()
        self.radiations = self.radiations.union({(1, i) for i in range(5)})
        self.radiations = self.radiations.union({(10, i) for i in range(5)})
        self.radiations = self.radiations.union({(11, i) for i in range(5)})

    def step(self, action) -> (object, [float, float], bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (time_inverted, treasure_value), final, info)
        """

        # (objective 1, objective 2, time)
        rewards = [0., 0., 0.]

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
            rewards[0] *= 2.
            rewards[1] *= 2.

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
