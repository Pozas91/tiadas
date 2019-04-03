"""
Such as Resource Gathering environment, but has a `time_limit`, if the agent non-reached goal in the `time_limit`, the
reward vector is divide by the `time` spent.
"""
import numpy as np

from .env_mesh import EnvMesh


class ResourceGatheringLimit(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Treasures
    _treasures = {'GOLD': 0, 'GEM': 1}

    def __init__(self, mesh_shape=(5, 5), initial_state=(2, 4), default_reward=0., seed=0, enemies=None,
                 gold_states=None, gem_states=None, p_attack=0.1, time_limit=100):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        """

        # [enemy_attack, gold, gems]
        self.state = [0, 0, 0]

        # States where there are gold
        if gold_states is None:
            # {state: available}
            gold_states = {(2, 0): True}

        self.gold_states = gold_states

        # States where there is a gem
        if gem_states is None:
            # {state: available}
            gem_states = {(4, 1): True}

        self.gem_states = gem_states

        # Time inverted in find a treasure
        self.time = 0
        self.time_limit = time_limit

        super().__init__(mesh_shape, seed, initial_state=initial_state, default_reward=default_reward)

        # States where there are enemies
        if enemies is None:
            enemies = [(3, 0), (2, 1)]

        self.enemies = enemies
        self.p_attack = p_attack

    def step(self, action) -> (object, [float, float, float], bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Final
        final = False

        # Calc rewards
        rewards = np.multiply(self.state, 0).tolist()

        # Get new state
        new_state = self._next_state(action=action)

        # Update previous state
        self.current_state = new_state
        self.time += 1

        if self.current_state in self.enemies:
            final = self.__enemy_attack()
            rewards = np.multiply(self.state, 1).tolist()
        elif self.current_state in self.gold_states.keys():
            self.__get_gold()
        elif self.current_state in self.gem_states.keys():
            self.__get_gem()
        elif self.__at_home():
            final = self.__is_checkpoint()
            rewards = np.multiply(self.state, 1).tolist()

        if self.time >= self.time_limit:
            final = True
            # Accumulate reward
            rewards = np.divide(self.state, self.time).tolist()

        # Set info
        info = {}

        return (self.current_state, tuple(self.state)), rewards, final, info

    def reset(self):
        """
        Reset environment to zero.
        :return:
        """
        self.current_state = self.initial_state
        self.state = np.multiply(self.state, 0).tolist()

        # Reset golds positions
        for gold_state in self.gold_states.keys():
            self.gold_states.update({gold_state: True})

        # Reset gems positions
        for gem_state in self.gem_states.keys():
            self.gem_states.update({gem_state: True})

        # Reset time inverted
        self.time = 0

        return self.current_state

    def render(self, **kwargs):
        # Get cols (x) and rows (y) from observation space
        cols, rows = self.observation_space.spaces[0].n, self.observation_space.spaces[1].n

        for y in range(rows):
            for x in range(cols):

                # Set a state
                state = (x, y)

                if state == self.current_state:
                    icon = self._icons.get('CURRENT')
                elif state in self.gold_states.keys():
                    icon = self._icons.get('TREASURE')
                elif state in self.gem_states.keys():
                    icon = self._icons.get('TREASURE')
                elif state in self.enemies:
                    icon = self._icons.get('ENEMY')
                elif state == self.initial_state:
                    icon = self._icons.get('HOME')
                else:
                    icon = self._icons.get('BLANK')

                # Show col
                print('| {} '.format(icon), end='')

            # New row
            print('|')

        # End render
        print('')

    def __enemy_attack(self) -> bool:
        """
        Check if enemy attack you
        :return:
        """

        final = False

        if self.p_attack >= self.np_random.uniform():
            self.reset()
            self.state[0] = -1
            final = True

        return final

    def __get_gold(self):
        """
        Check if agent can take the gold.
        :return:
        """

        # Check if there is a gold
        if self.gold_states.get(self.current_state, False):
            self.state[1] += 1
            self.gold_states.update({self.current_state: False})

    def __get_gem(self):
        """
        Check if agent can take the gem.
        :return:
        """

        # Check if there is a gem
        if self.gem_states.get(self.current_state, False):
            self.state[2] += 1
            self.gem_states.update({self.current_state: False})

    def __at_home(self) -> bool:
        """
        Check if agent is at home
        :return:
        """

        return self.current_state == self.initial_state

    def __is_checkpoint(self) -> bool:
        """
        Check if is final state (has gold, gem or both)
        :return:
        """

        return (self.state[1] >= 0 or self.state[2] >= 0) and self.__at_home()
