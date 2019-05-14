"""
Variant of the Deep Sea Treasure environment.
"""
from .deep_sea_treasure import DeepSeaTreasure


class DeepSeaTreasureMixed(DeepSeaTreasure):

    def __init__(self, initial_state: tuple = (0, 0), default_reward: tuple = (0,), seed: int = 0):
        """
        :param initial_state:
        :param default_reward:
        :param seed:
        """

        super().__init__(initial_state=initial_state, default_reward=default_reward, seed=seed)

        # List of all treasures and its reward.
        self.finals = {
            (0, 1): 1,
            (1, 2): 2,
            (2, 3): 10,
            (3, 4): 11,
            (4, 4): 12,
            (5, 4): 13,
            (6, 7): 15,
            (7, 7): 18.5,
            (8, 9): 19,
            (9, 10): 20,
        }

        # Default reward plus time (time_inverted, treasure_value)
        default_reward = (-1,) + default_reward
        self.default_reward = VectorFloat(default_reward)

    def step(self, action: int) -> (tuple, VectorFloat, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return: (state, (time_inverted, treasure_value), final, info)
        """

        # Initialize rewards as vector (plus zero to fast copy)
        rewards = self.default_reward + 0

        # Get new state
        new_state = self.next_state(action=action)

        # Update previous state
        self.current_state = new_state

        # Get treasure value
        rewards[1] = self.finals.get(self.current_state, self.default_reward[1])

        # Set info
        info = {}

        # Check is_final
        final = self.is_final(self.current_state)

        return self.current_state, rewards, final, info
