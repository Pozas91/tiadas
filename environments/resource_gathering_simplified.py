"""
An agent begins at the home location in a 2D grid, and can move one square at a time in each of the four cardinal
directions. The agent's task is to collect either or both of two resources (gold and gems) which are available at fixed
locations, and return home with these resources. The environment contains two locations at which an enemy attack may
occur, with a 10% probability. If an attack happens, the agent loses any resources currently being carried and is
returned to the home location. The reward vector is ordered as [enemy, gold, gems] and there are four possible rewards
which may be received on entering the home location.

• [−1, 0, 0] in case of an enemy attack;
• [0, 1, 0] for returning home with gold but no gems;
• [0, 0, 1] for returning home with gems but no gold;
• [0, 1, 1] for returning home with both gold and gems.

FINAL STATE: Doesn't have final state. Continuous task.

REF: Empirical Evaluation methods for multi-objective reinforcement learning algorithms
    (Vamplew, Dazeley, Berry, Issabekov and Dekker) 2011
"""

from . import ResourceGathering


class ResourceGatheringSimplified(ResourceGathering):

    def __init__(self, initial_state: tuple = ((1, 2), (0, 0)), default_reward: tuple = (0, 0, 0), seed: int = 0,
                 p_attack: float = 0.1, mesh_shape: tuple = (4, 3), gold_positions: frozenset = frozenset({1, 0}),
                 gem_positions: frozenset = frozenset({(3, 1)})):
        """
        :param initial_state:
        :param default_reward: (enemy_attack, gold, gems)
        :param seed:
        :param p_attack: Probability that a enemy attacks when agent stay in an enemy position.
        """

        # Super constructor call.
        super().__init__(initial_state=initial_state, default_reward=default_reward, seed=seed, p_attack=p_attack,
                         mesh_shape=mesh_shape, gold_positions=gold_positions, gem_positions=gem_positions)

        self.checkpoints_states = {
            ((1, 2), (1, 0)),
            ((1, 2), (0, 1)),
            ((1, 2), (1, 1)),
        }

        # States where there are enemies_positions
        self.enemies_positions = {(2, 0), (1, 1)}
        self.home_position = (1, 2)

    def warning_action(self, state: tuple, action: int):
        return ((state[0] == (2, 1) or state[0] == (2, 0)) and action == self.actions['UP']) or \
               (state[0] == (2, 1) and action == self.actions['LEFT']) or \
               (state[0] == (3, 0) and action == self.actions['LEFT']) or \
               (state[0] == (1, 2) and action == self.actions['UP']) or \
               (state[0] == (0, 1) and action == self.actions['RIGHT']) or \
               (state[0] == (1, 0) and action == self.actions['DOWN']) or \
               (state[0] == (1, 0) and action == self.actions['RIGHT'])

    def reachable_states(self, state: tuple, action: int) -> set:

        reachable_states = set()

        if (state[0] == (2, 1) or state[0] == (2, 0)) and action == self.actions['UP']:
            reachable_states.add(((2, 0), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (2, 1) and action == self.actions['LEFT']:
            reachable_states.add(((1, 1), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (3, 0) and action == self.actions['LEFT']:
            reachable_states.add(((2, 0), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (1, 2) and action == self.actions['UP']:
            reachable_states.add(((1, 1), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (0, 1) and action == self.actions['RIGHT']:
            reachable_states.add(((1, 1), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (1, 0) and action == self.actions['DOWN']:
            reachable_states.add(((1, 1), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        elif state[0] == (1, 0) and action == self.actions['RIGHT']:
            reachable_states.add(((2, 0), state[1]))
            reachable_states.add((self.home_position, (0, 0)))
        else:
            reachable_states.add(self.next_state(action=action, state=state))

        # Return all possible states reachable with any action
        return reachable_states
