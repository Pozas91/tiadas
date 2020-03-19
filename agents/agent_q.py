"""
Q-Learning agent to resolve environments trough reinforcement learning.

The train_data structure of q dictionary is as follows:

{
    state_1: {action_1: reward, action_2: reward, action_3: reward, ...},
    state_2: {action_1: reward, action_2: reward, action_3: reward, ...},
    state_3: {action_1: reward, action_2: reward, action_3: reward, ...},
    ...
}
"""
from copy import deepcopy
from pprint import pprint

import numpy as np

import utils.numbers as un
from environments import Environment
from models import GraphType, VectorDecimal
from .agent_rl import AgentRL


class AgentQ(AgentRL):
    def __init__(self, environment: Environment, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 1.,
                 seed: int = 0, states_to_observe: set = None, max_steps: int = None, graph_types: set = None,
                 initial_value: float = 0.):
        """
        :param environment: An environment where agent does any operation.
        :param alpha: Learning rate
        :param epsilon: Epsilon using in e-greedy policy, to explore more states.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param states_to_observe: List of states from that we want to get a graphical output.
        :param max_steps: Limits of steps per episode.
        :param graph_types: Types of graphs to generate.
        :param initial_value: Value with the algorithm begin to learn (by default zero).
        """

        # Types to make graphs
        if graph_types is None:
            graph_types = {GraphType.EPISODES, GraphType.STEPS}

        # Super call __init__
        super().__init__(environment=environment, epsilon=epsilon, gamma=gamma, seed=seed,
                         initial_value=initial_value, states_to_observe=states_to_observe, max_steps=max_steps,
                         graph_types=graph_types)

        # Learning factor
        assert 0 < alpha <= 1
        self.alpha = alpha

        # Initialize to Q-Learning values
        self.q = dict()

        # Rewards history train_data
        self.rewards_history = list()

    def walk(self, from_state: object = None) -> list:
        """
        Do a walk follows best current policy
        :return:
        """

        # Reset mesh
        self.state = self.environment.reset()

        # Check if other initial position is selected
        if from_state:
            self.state = from_state

        # Condition to stop walk
        is_final_state = False

        # Reset steps
        self.reset_steps()

        # Rewards history
        history = list()

        while not is_final_state:
            # Increment steps
            self.steps += 1

            # Get an action
            action = self._best_action()

            # Do step on environment
            next_state, reward, is_final_state, info = self.environment.step(action=action)

            # Append to rewards history
            history.append(reward)

            # Update position
            self.state = next_state

            # Check timeout
            if self.max_steps is not None and not is_final_state:
                is_final_state = self.steps >= self.max_steps

        return history

    def do_step(self) -> bool:
        """
        The agent does a step to learn vectors.
        :return:
        """

        # Get an action
        action = self.select_action()

        # Do step on environment
        next_state, reward, is_final_state, info = self.environment.step(action=action)

        # Increment steps
        self.total_steps += 1
        self.steps += 1

        # Append to rewards history
        self.rewards_history.append(reward)

        # Processing reward
        reward = self.process_reward(reward=reward)

        # Update Q-Values
        self._update_q_values(reward=reward, action=action, next_state=next_state)

        # Update position
        self.state = next_state

        return is_final_state

    def update_graph(self, graph_type: GraphType):
        """
        Update specific graph type
        :param graph_type:
        :return:
        """

        for state, data in self.graph_info[graph_type].items():
            # Add to train_data Best value (V max)
            value = self._best_reward(state=state)

            # Add to train_data Best value (V max)
            data.append(value)

            # Update dictionary
            self.graph_info[graph_type].update({state: data})

    def _update_q_values(self, reward: float, action: int, next_state: object) -> None:
        """
        Update Q-Dictionary with new train_data
        :param reward:
        :param action:
        :param next_state:
        :return:
        """

        # Get old value
        old_value = self.q.get(self.state, {}).get(action, 0.)

        # Get next max value
        next_max = self._best_reward(state=next_state)

        # Calc new value apply Q-Learning formula:
        # Q(St, At) <- (1 - alpha) * Q(St, At) + alpha * (r + y * Q(St_1, action))
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + next_max * self.gamma)

        # Prepare new train_data
        new_data = {action: new_value}

        # If we know this position
        if self.state in self.q:
            # Update value for the action done.
            self.q.get(self.state).update(new_data)
        else:
            # Set a new dictionary for this position
            self.q.update({self.state: new_data})

    def show_q(self) -> None:
        """
        Show Q-Data
        :return:
        """
        pprint(self.q)

    def show_policy(self) -> None:
        """
        Show all states with it'state best action
        :return:
        """
        # For each position in q
        for state in self.q.keys():
            best_action = self._best_action(state=state)
            print("State: {} -> Action: {}".format(state, best_action))

    def reset(self) -> None:
        """
        Reset agent, forgetting previous q-values
        :return:
        """
        # Super call to reset method
        super().reset()

        self.rewards_history = list()
        self.q = dict()
        self.state = self.environment.reset()
        self.steps = 0

    def _best_action(self, state: object = None, extra: object = None) -> int:
        """
        Return best action for q and position given.
        :return:
        """

        # if don't specify a position, get current position.
        if state is None:
            state = self.state

        # Get information about all actions with its rewards.
        possible_actions = self.q.get(state, {})

        # Get unknown actions with default reward
        for action in self.environment.action_space:
            if action not in possible_actions:
                possible_actions.update({action: self.initial_q_value})

        # Get max by value, and get it'state action
        actions = list()
        max_reward = float('-inf')

        # Check all actions with it'state rewards
        for possible_action in possible_actions:

            # Get current Value
            reward = possible_actions.get(possible_action)

            # If current value is close to new value
            if un.are_equal_two_decimal_numbers(a=reward, b=max_reward):

                # Append another possible action
                actions.append(possible_action)

            # If current value is best than old value
            elif reward > max_reward:

                # Create a new list with current key.
                actions = [possible_action]

            # Update max value
            max_reward = max(max_reward, reward)

        # From best actions get one aleatory.
        action = self.generator.choice(actions)

        return action

    def _best_reward(self, state: object) -> float:
        """
        Return best reward for q and position given
        :return:
        """

        # If this position is a final position, then reward is zero.
        if self.environment.is_final(state):
            reward = 0.0
        else:

            # Get information about possible actions
            possible_actions = self.q.get(state, {})

            # Get unknown actions with default reward
            for action in self.environment.action_space:
                if action not in possible_actions:
                    possible_actions.update({action: self.initial_q_value})

            # Get best action and use it to get best reward.
            action = self._best_action(state=state)
            reward = possible_actions.get(action)

        return reward

    @property
    def v(self) -> float:
        """
        Get best value from initial position -> V_max(0, 0)
        :return:
        """
        return self._best_reward(state=self.environment.initial_state)

    def reset_rewards_history(self) -> None:
        """
        Forget rewards history
        :return:
        """
        self.rewards_history = list()

    def process_reward(self, reward: float) -> float:
        """
        Processing reward function.
        :param reward:
        :return:
        """
        return reward

    def show_v_values(self) -> None:
        """
        Show Best rewards from Q-Dictionary
        :return:
        """
        for state in self.q.keys():
            # Get rewards
            rewards = self.q.get(state).values()

            # Print result
            print('State: {} -> V: {}'.format(state, max(rewards)))

    def objective_training(self, objective: float, graph_type: GraphType = None) -> None:
        """
        Train until agent V(0, 0) value is close to objective value.
        :param graph_type:
        :param objective:
        :return:
        """

        while not un.are_equal_two_decimal_numbers(a=self.v, b=objective):
            # Do an episode
            self.episode(graph_type=graph_type)

    def exhaustive_train(self, graph_type: GraphType = None) -> None:
        """
        Train until Agent is stabilized
        :return:
        """

        # Initialize variables
        q_previous = dict()
        steps = 0
        steps_margin = 0

        # First check
        are_similar = self.policies_are_similar(q_previous, self.q)

        while not are_similar or steps_margin < 20:

            # Get previous Q-Values
            q_previous = deepcopy(self.q)

            # Do an episode
            self.episode(graph_type=graph_type)

            # Increment steps
            steps += 1

            # Check again
            are_similar = self.policies_are_similar(q_previous, self.q)

            # Control false positive
            if are_similar:
                steps_margin += 1
            else:
                steps_margin = 0

    def get_accumulated_reward(self, from_state: object = None) -> VectorDecimal:
        """
        When the agent is trained, do a walk, and return the sum of vectors recovered.
        :return:
        """
        self.state = self.environment.reset()

        # Get history of walk
        history = self.walk(from_state=from_state)

        # Sum history to get total reward
        result = np.sum(history, axis=0)

        # Return a vector float
        return VectorDecimal(result)

    @staticmethod
    def policies_are_similar(a: dict, b: dict) -> bool:
        """
        Check if two policies are similar
        :param a:
        :param b:
        :return:
        """
        a_states = list(a.keys())
        b_states = list(b.keys())

        # Must be same keys
        same_keys = a_states == b_states

        # Neither dictionary is empty
        neither_is_empty = bool(a) and bool(b)

        are_similar = True

        # If have same keys and neither is empty
        if same_keys and neither_is_empty:
            i = 0
            len_a_states = len(a_states)

            while i < len_a_states and are_similar:

                # Get a position
                state = a_states[i]

                # Get actions from dictionaries
                a_actions = a.get(state)
                b_actions = b.get(state)

                a_actions_keys = list(a_actions.keys())
                b_actions_keys = list(b_actions.keys())

                # Prepare while loop
                j = 0
                len_a_actions = len(a_actions_keys)

                # Confirm that a_actions and b_actions are equals
                are_similar &= a_actions_keys == b_actions_keys

                while j < len_a_actions and are_similar:
                    # Get an action
                    action = a_actions_keys[j]

                    # Get and compare if both values are similar
                    a_value = a.get(state).get(action)
                    b_value = b.get(state).get(action, float('inf'))
                    are_similar &= a_value.all_close(v2=b_value)

                    # Increment j
                    j += 1

                # Increment i
                i += 1
        else:
            are_similar = False

        return are_similar
