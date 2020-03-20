"""
A variant of Space Exploration environment for acyclic agents.

Remove all rewards from left side, start at (0, 0) position, and remove some asteroids states. We can go only to RIGHT,
DOWN RIGHT and DOWN

HV REFERENCE: (-100, -150)
"""
import utils.environments as ue
from environments import SpaceExploration
from spaces import Bag


class SpaceExplorationAcyclic(SpaceExploration):
    # Possible actions
    _actions = {'RIGHT': 0, 'DOWN RIGHT': 1, 'DOWN': 2}

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0, -1), seed: int = 0):
        """
        :param initial_state: Initial state where start the agent.
        :param default_reward: (mission_success, radiation)
        :param seed: Seed used for np.random.RandomState method.
        """

        action_space = Bag([])
        action_space.seed(seed=seed)

        super().__init__(initial_state=initial_state, default_reward=default_reward, seed=seed,
                         action_space=action_space)

        # List of all treasures and its reward.
        self.finals = {}
        self.finals.update({(9, i): 20 for i in range(3)})
        self.finals.update({(12, i): 30 for i in range(5)})

        # Define asteroids states
        self.asteroids = {
            (5, 0), (5, 4),
            (6, 3), (6, 1),
            (7, 2)
        }

        # Define radiation states
        self.radiations = set()
        self.radiations = self.radiations.union({(10, i) for i in range(5)})
        self.radiations = self.radiations.union({(11, i) for i in range(5)})

    def next_position(self, action: int, position: tuple) -> (tuple, bool):
        """
        Given an action and a position, return the next position reached.
        :param action:
        :param position:
        :return:
        """
        # Get my position
        x, y = position

        # Get observations spaces
        observation_space_x, observation_space_y = self.observation_space.spaces

        # Do movement in cyclic mesh
        if action == self.actions['RIGHT']:
            x = ue.move_right(x=x, limit=observation_space_x.n)
        elif action == self.actions['DOWN']:
            y = ue.move_down(y=y, limit=observation_space_y.n)
        elif action == self.actions['DOWN RIGHT']:
            y = ue.move_down(y=y, limit=observation_space_y.n)
            x = ue.move_right(x=x, limit=observation_space_x.n)

        # Set new position
        next_position = x, y

        return next_position, True

    def next_state(self, action: int, state: tuple = None) -> tuple:
        """
        Calc next position with current position and action given, in this environment is 8-neighbors.
        :param state: If a position is given, do action from that position.
        :param action: from action_space
        :return:
        """

        # Get my position
        position = state if state else self.current_state

        next_position, is_valid = self.next_position(action=action, position=position)

        if not self.observation_space.contains(next_position) or not is_valid:
            raise ValueError('Action/state combination produces cycles.')

        # Return (x, y) position
        return next_position

    @property
    def action_space(self) -> Bag:
        """
        Get a dynamic action space with only valid actions.
        :return:
        """

        # Get current position
        x, y = self.current_state

        # Get observations spaces
        observation_space_x, observation_space_y = self.observation_space.spaces

        # Setting possible actions
        possible_actions = []

        # Can we go to RIGHT?
        x_right = ue.move_right(x=x, limit=observation_space_x.n)

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x_right, y)):
            # We can go to right
            possible_actions.append(self.actions['RIGHT'])

        # Can we go to DOWN RIGHT?
        x_right = ue.move_right(x=x, limit=observation_space_x.n)
        y_down = ue.move_down(y=y, limit=observation_space_y.n)

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x_right, y_down)):
            # We can go to right
            possible_actions.append(self.actions['DOWN RIGHT'])

        # Can we go to DOWN?
        y_down = y + 1

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x, y_down)):
            # We can go to right
            possible_actions.append(self.actions['DOWN'])

        # Setting to dynamic_space
        self._action_space.items = possible_actions

        # Update n length
        self._action_space.n = len(possible_actions)

        # Return a list of iterable valid actions
        return self._action_space
