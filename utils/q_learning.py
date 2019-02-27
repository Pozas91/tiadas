from copy import deepcopy

import numpy as np

from agents import Agent


def train(agent: Agent, epochs=int(1e4), verbose=False):
    """
    Return an agent trained with `epochs` epochs.
    :param verbose:
    :param agent:
    :param epochs:
    :return:
    """

    for _ in range(epochs):
        # Do an episode
        agent.episode()

    # Show values
    if verbose:
        agent.show_q()


def cheat_train(agent: Agent, objective: float, close_margin=1e-9):
    # Initialize variable
    v = 0

    while not is_close(a=v, b=objective, relative=close_margin):
        # Do an episode
        agent.episode()

        # Get v(0, 0)
        v = agent.get_v()


def exhaustive_train(agent: Agent, close_margin=1e-3):
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
                are_similar &= is_close(a=a_value, b=b_value, relative=close_margin)

                # Increment j
                j += 1

            # Increment i
            i += 1
    else:
        are_similar = False

    return are_similar


def normalized(data: list) -> list:
    data = np.array(data)
    return (data / np.linalg.norm(data)).tolist()


def testing(agent: Agent) -> list:
    """
    Test policy
    :param agent:
    :return:
    """
    initial_state = agent.environment.reset()
    agent.__setstate__(initial_state)

    history = agent.walk()

    return history[-1]


def is_close(a: float, b: float, relative=1e-3):
    return abs(a - b) <= relative
