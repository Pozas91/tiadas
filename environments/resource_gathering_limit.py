"""
Such as Resource Gathering environment, but has a `time_limit`, if the agent non-reached goal in the `time_limit`, the
reward vector is divide by the `time` spent.
"""
from environments import ResourceGathering
from models import VectorDecimal


class ResourceGatheringLimit(ResourceGathering):

    def __init__(self, initial_state: tuple = ((2, 4), (0, 0)), default_reward: tuple = (0, 0, 0), seed: int = 0,
                 p_attack: float = 0.1, time_limit: int = 100):
        """
        :param initial_state:
        :param default_reward: (enemy_attack, gold, gems)
        :param seed:
        :param p_attack: Probability that a enemy attacks when agent stay in an enemy position.
        :param time_limit: When agent does `time_limit` steps terminate current episode.
        """

        # Time inverted in find a treasure
        self.time = 0
        self.time_limit = time_limit

        super().__init__(initial_state=initial_state, default_reward=default_reward, seed=seed, p_attack=p_attack)

    def step(self, action: int) -> (tuple, VectorDecimal, bool, dict):
        """
        Given an action, do a step
        :param action:
        :return:
        """

        # Call super step method
        next_state, reward, final, info = super().step(action=action)

        # Increment time
        self.time += 1

        if self.time >= self.time_limit:

            # Prepare reward
            objects = list(self.current_state[1])
            objects.insert(0, 0)

            # Accumulate reward
            reward = list(map(lambda x: x / self.time, objects))
            final = True

        return next_state, reward, final, info

    def reset(self) -> tuple:
        """
        Reset environment to zero.
        :return:
        """

        # Super call reset method
        self.current_state = super().reset()

        # Reset time inverted
        self.time = 0

        return self.current_state
