"""
Variant of the Deep Sea Treasure environment.
"""
from models import VectorDecimal
from .deep_sea_treasure import DeepSeaTreasure


class DeepSeaTreasureMixed(DeepSeaTreasure):
    # Pareto optimal
    pareto_optimal = [
        (-1, 1), (-3, 2), (-5, 10), (-7, 11), (-8, 12), (-9, 13), (-13, 15), (-14, 18.5), (-17, 19), (-19, 20)
    ]

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
        self.default_reward = VectorDecimal(self.default_reward)
