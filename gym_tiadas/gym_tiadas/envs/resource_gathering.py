"""
An agent begins at the home location in a 2D grid, and can move one square at a time in each of the four cardinal
directions. The agent's task is to collect either or both of two resources (gold and gems) which are available at fixed
locations, and return home with these resources. The environment contains two locations at which an enemy attack may
occur, with a 10% probability. If an attack happens, the agent loses any resources currently being carried and is returned
to the home location. The reward vector is ordered as [enemy, gold, gems] and there are four possible rewards which may
be received on entering the home location.

• [−1, 0, 0] in case of an enemy attack;
• [0, 1, 0] for returning home with gold but no gems;
• [0, 0, 1] for returning home with gems but no gold;
• [0, 1, 1] for returning home with both gold and gems.

FINAL STATE: any of below states.

REF: Empirical Evaluation methods for multi-objective reinforcement learning algorithms
    (Vamplew, Dazeley, Berry, Issabekov and Dekker) 2011
"""
import numpy as np

from .env_mesh import EnvMesh


class ResourceGathering(EnvMesh):
    # Possible actions
    _actions = {'UP': 0, 'RIGHT': 1, 'DOWN': 2, 'LEFT': 3}

    # Treasures
    _treasures = {'GOLD': 0, 'GEM': 1}

    def __init__(self, initial_state=(2, 4), default_reward=0., seed=0, p_attack=0.1):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        """

        # [enemy_attack, gold, gems]
        self.state = [0, 0, 0]

        # States where there are gold {state: available}
        self.gold_states = {(2, 0): True}

        # States where there is a gem {state: available}
        self.gem_states = {(4, 1): True}

        mesh_shape = (5, 5)

        # Super constructor call.
        super().__init__(mesh_shape, seed, initial_state=initial_state, default_reward=default_reward)

        # States where there are enemies
        self.enemies = [(3, 0), (2, 1)]

        self.p_attack = p_attack

    def step(self, action) -> (object, [float, float, float], bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Calc rewards
        rewards = np.multiply(self.state, 0).tolist()

        # Get new state
        new_state = self._next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Check is_final
        final = self.is_final(self.current_state)

        # If the agent is in the same state as an enemy
        if self.current_state in self.enemies:
            rewards = np.multiply(self.state, 1).tolist()

        # If the agent is in the same state as gold
        elif self.current_state in self.gold_states.keys():
            self.__get_gold()

        # If the agent is in the same state as gem
        elif self.current_state in self.gem_states.keys():
            self.__get_gem()

        # If the agent is at home and have gold or gem
        elif self.__at_home():
            rewards = np.multiply(self.state, 1).tolist()

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

    def is_final(self, state=None) -> bool:
        """
        Is final if the agent is attacked or agent is on checkpoint.
        :param state:
        :return:
        """
        # Check if agent is attacked
        attacked = state in self.enemies and self.__enemy_attack()
        # Check if agent is on checkpoint
        checkpoint = state == self.initial_state and self.__is_checkpoint()

        return attacked or checkpoint

    def get_dict_model(self):
        """
        Get dict model of environment
        :return:
        """

        data = super().get_dict_model()

        # Clean specific environment data
        del data['gold_states']
        del data['gem_states']
        del data['enemies']

        return data
