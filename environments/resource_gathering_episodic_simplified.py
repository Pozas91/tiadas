"""
Episodic version of ResourceGathering.

REF: 'A temporal difference method for multi-objective reinforcement learning'
    (Manuela Ruíz-Montiel, Lawrence Mandow, José-Luis Pérez-de-la-Cruz 2017)

FINAL STATES: Return at home with any object or attacked by enemy.
"""

from models import Vector
from .resource_gathering_simplified import ResourceGatheringSimplified


class ResourceGatheringEpisodicSimplified(ResourceGatheringSimplified):

    def __init__(self, initial_state: tuple = ((1, 2), (0, 0)), default_reward: tuple = (0, 0, 0), seed: int = 0,
                 p_attack: float = 0.1, steps_limit: int = 1000):
        """
        :param initial_state:
        :param default_reward: (enemy_attack, gold, gems)
        :param seed:
        :param p_attack: Probability that a enemy attacks when agent stay in an enemy position.
        """

        super().__init__(initial_state=initial_state, default_reward=default_reward, seed=seed, p_attack=p_attack)

        # Define final states
        self.finals = self.checkpoints_states.copy()

        self.steps = 0
        self.steps_limit = steps_limit

    def step(self, action: int) -> (tuple, Vector, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Increment steps
        self.steps += 1

        # Call super method
        self.current_state, reward, final, info = super().step(action=action)

        # Final timeout
        if final and self.steps >= self.steps_limit:
            reward[1], reward[2] = 0, 0

        return self.current_state, reward, final, info

    def reset(self) -> tuple:
        """
        Reset environment to zero.
        :return:
        """

        # Super method call
        super().reset()

        # Reset steps
        self.steps = 0

        return self.current_state

    def transition_reward(self, state: tuple, action: int, next_state: tuple) -> Vector:

        # Super method call
        reward = super().transition_reward(state=state, action=action, next_state=next_state)

        # Final timeout
        if self.steps >= self.steps_limit:
            reward[1], reward[2] = 0, 0

        return reward

    def is_final(self, state: tuple = None) -> bool:
        """
        Return True if position given is terminal, agent was attacked or steps_limit was exceeded, False in otherwise.
        :return:
        """
        return super().is_final(state=state) or self.steps >= self.steps_limit or self.attacked
