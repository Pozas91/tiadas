from environments import PyramidMDP
from spaces import Bag


class PyramidMDPNoBounces(PyramidMDP):

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (-1, -1), seed: int = 0,
                 n_transition: float = 0.95, diagonals: int = 9):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        :param n_transition: if is 1, always do the action indicated. (Original is about 0.6)
        """

        # Action space
        action_space = Bag([])
        action_space.seed(seed=seed)

        super().__init__(initial_state=initial_state, default_reward=default_reward, seed=seed, diagonals=diagonals,
                         n_transition=n_transition, action_space=action_space)

    @property
    def action_space(self) -> Bag:
        """
        Get a dynamic action space with only valid actions.
        :return:
        """

        # Get current position
        x, y = self.current_state

        # Setting possible actions
        possible_actions = []

        # Can we go to right?
        x_right = x + 1

        # Check that x_right is not an obstacle and is into mesh
        if self.observation_space.contains((x_right, y)):
            # We can go to right
            possible_actions.append(self.actions['RIGHT'])

        # Can we go to up?
        y_up = y - 1
        if self.observation_space.contains((x, y_up)):
            # We can go to up
            possible_actions.append(self.actions['UP'])

        # Can we go to left?
        x_left = x - 1
        if self.observation_space.contains((x_left, y)):
            # We can go to left
            possible_actions.append(self.actions['LEFT'])

        # Can we go to left?
        y_down = y + 1
        if self.observation_space.contains((x, y_down)):
            # We can go to left
            possible_actions.append(self.actions['DOWN'])

        # Setting to dynamic_space
        self._action_space.items = possible_actions

        # Update n length
        self._action_space.n = len(possible_actions)

        # Return a list of iterable valid actions
        return self._action_space

    def reachable_states(self, state: tuple, action: int) -> set:

        # Set current state with state indicated
        self.current_state = state

        # Get all actions available
        actions = self.action_space.copy()

        # Unpack position
        x, y = state

        if (x <= 0 and self.actions['LEFT'] in actions) or (y <= 0 and self.actions['UP'] in actions):
            raise ValueError('Action/State combination returns a cyclic position.')

        # Return all possible states reachable with any action
        return {self.next_state(action=a, state=state) for a in actions}
