from agents import Agent, AgentMultiObjective


def train(agent: AgentMultiObjective, epochs=int(1e5), verbose=False):
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
