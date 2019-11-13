import time
from itertools import product
from pathlib import Path
from typing import Iterator

import numpy as np

from environments import Environment
from models import VectorDecimal


class AgentSimulation:

    def __init__(self, environment: Environment):
        self.environment = environment

    def policies(self, reverse: bool = False) -> Iterator:
        """
        Extract all possible policies exhaustively.
        :return:
        """
        possible_actions = list()

        states = self.environment.ordered_states(reverse=reverse)

        for state in states:
            self.environment.current_state = state
            possible_actions.append(self.environment.action_space.items)

        return product(*possible_actions), states

    def simulate(self, policy: tuple, states: list) -> np.ndarray:
        """
        Simulate the given policy until the value converge with `Vector.decimals_allowed` decimals.
        `states` list is a list of ordered states, necessary to know the state position in the policy tuple.
        :param policy:
        :param states:
        :return:
        """

        rewards = list()
        are_close = False
        seed = 0
        last_avg = VectorDecimal(self.environment.default_reward.zero_vector)

        while not are_close:

            # Reset environment
            self.environment.initial_seed = seed
            self.environment.reset()

            accumulated_reward = self.environment.default_reward.zero_vector

            is_final_state = False

            while not is_final_state:
                i_state = states.index(self.environment.current_state)
                next_state, reward, is_final_state, _ = self.environment.step(action=policy[i_state])
                accumulated_reward += reward

            rewards.append(accumulated_reward)

            # Check if current average is similar to last average
            current_avg = VectorDecimal(np.average(rewards, axis=0))
            are_close = current_avg.all_close(last_avg)
            last_avg = current_avg.copy()

            seed += 1

        print('Iterations: {}'.format(len(rewards)))

        return np.average(rewards, axis=0)

    def policies_avg_reward(self, reverse: bool = False):
        """
        Calculate the average reward for each policy.
        :return:
        """

        policies, states = self.policies(reverse=reverse)
        policies_avg_reward = dict()

        for policy in policies:
            # Calc average reward and set to the policy
            avg_reward = self.simulate(policy=policy, states=states)

            # Update policies average reward dictionary
            policies_avg_reward.update({policy: avg_reward})

        # Return policies with its average reward associate
        return policies_avg_reward

    @staticmethod
    def dumps_policies_avg_reward(policies_avg_reward: dict):
        """
        Save all policies with it reward in a dumps file.
        :param policies_avg_reward:
        :return:
        """

        timestamp = int(time.time())

        # Specify full path
        file_path = Path(__file__).parent.parent.joinpath(
            'dumps/simulation/train_data/dstrds_{}.yml'.format(timestamp)
        )

        # If any parents doesn't exist, make it.
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open(mode='w+', encoding='UTF-8') as f:
            file_data = 'v_s_0 = {}'.format([tuple(v.tolist()) for v in policies_avg_reward.values()])
            f.write(file_data)
