from agents import Agent
import numpy as np


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


def normalized(data: list) -> list:
    data = np.array(data)
    return (data / np.linalg.norm(data)).tolist()
