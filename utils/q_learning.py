"""
Useful functions to calculate a reinforcement learning based on q-learning. This file has three ways to train an agent:

    * train
    * objective_train
    * exhaustive_train

EXAMPLE OF USE TO TRAIN METHOD:

    # Define environment
    environment = SpaceExploration()

    # Prepare instance of agent
    agent = AgentMOSP(environment=environment, weights=[0.8, 0.2], epsilon=0.5, alpha=0.2, states_to_observe=[(0, 0)])

    # Train agent (THIS IS THE CALL OF THESE FUNCTIONS)
    q_learning.train(agent=agent, epochs=100000)

    # Show policy learned
    agent.show_policy()


EXAMPLE OF USE TO OBJECTIVE METHOD:

    # Reset agent (forget q-values, initial_state, etc.).
    agent.reset()

    # Set news weights to get the new solution.
    agent.weights = [w1, w2]

    # If solutions not is None
    if solutions_known:
        # Multiply and sum all points with agent's weights.
        objectives = np.sum(np.multiply(solutions_known, [w1, w2]), axis=1)

        # Get max of these sums (That is the objective).
        objective = np.max(objectives)

        # Train agent searching that objective. (TRAIN UNTIL AGENT.V IS CLOSE TO OBJECTIVE)
        q_learning.objective_training(agent=agent, objective=objective, close_margin=3e-1)
    else:
        # Normal training.
        q_learning.train(agent=agent)

    # Get point c from agent's test.
    c = q_learning.testing(agent=agent)

    return c


EXAMPLE OF USE TO EXHAUSTIVE TRAIN METHOD:

    # Define environment
    environment = DeepSeaTreasure()

    # Prepare instance of agent
    agent = AgentMOSP(environment=environment, weights=[0.8, 0.2], epsilon=0.5, alpha=0.2, states_to_observe=[(0, 0)])

    # Train agent (TRAIN UNTIL POLICY DON'T CHANGE MORE, IS IMPORTANT THAT AGENT HAS `V` PROPERTY DEFINE)
    q_learning.exhaustive_train(agent)

    # Show results
    agent.print_observed_states()

"""

from copy import deepcopy

import numpy as np

import utils.miscellaneous as um
from models import Vector


def train(agent, epochs=int(1e3)):
    """
    Return an agent trained with `epochs` epochs.
    :param agent:
    :param epochs:
    :return:
    """

    for _ in range(epochs):
        # Do an episode
        agent.episode()


def objective_training(agent, objective: Vector, close_margin=1e-9):
    """
    Train until agent V(0, 0) value is close to objective value.
    :param agent:
    :param objective:
    :param close_margin:
    :return:
    """
    while not um.is_close(a=agent.v, b=objective, relative=close_margin):
        # Do an episode
        agent.episode()


def exhaustive_train(agent, close_margin=1e-3):
    """
    Train until Agent is stabilized
    :param close_margin: Margin of difference between two float numbers.
    :param agent:
    :return:
    """

    # Initialize variables
    q_previous = dict()
    iterations = 0
    iterations_margin = 0

    # First check
    are_similar = policies_are_similar(q_previous, agent.q, close_margin)

    while not are_similar or iterations_margin < 20:

        # Get previous Q-Values
        q_previous = deepcopy(agent.q)

        # Do an episode
        agent.episode()

        # Increment iterations
        iterations += 1

        # Check again
        are_similar = policies_are_similar(q_previous, agent.q, close_margin)

        # Control false positive
        if are_similar:
            iterations_margin += 1
        else:
            iterations_margin = 0

    print(iterations)


def policies_are_similar(a: dict, b: dict, close_margin=1e-3) -> bool:
    """
    Check if two policies are similar
    :param a:
    :param b:
    :param close_margin:
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

            # Get a state
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
                are_similar &= um.is_close(a=a_value, b=b_value, relative=close_margin)

                # Increment j
                j += 1

            # Increment i
            i += 1
    else:
        are_similar = False

    return are_similar


def testing(agent):
    """
    Test policy
    :param agent:
    :return:
    """
    agent.state = agent.environment.reset()

    # Get history of walk
    history = agent.walk()

    # Sum history to get total reward
    result = np.sum(history, axis=0)

    # Return a tuple of that sum
    return result
