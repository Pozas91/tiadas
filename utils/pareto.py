"""
Methods to calculate pareto's frontier applies dichotomous search to find supported solutions.
"""
import numpy as np

import utils.miscellaneous as um
import utils.q_learning as uq


def optimize(agent, w1: float, w2: float, solutions_known=None):
    """
    Try to find an c point to add to the pareto's frontier. There are two options:
        * We know the solutions of pareto's frontier, and training the agent until get max solution.
        * We don't know the solutions and training to try get max solution.

    :param agent: Must be an scalarized agent multi-objective single-policy (models.agent_mo_sp.AgentMOSP).
    :param solutions_known: If we know the possible solutions, we can indicate them to the algorithm to improve the
        training of the agent. If is None, then is ignored.
    :param w1: first weight
    :param w2: second weight
    :return:
    """

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
        # Train agent searching that objective.
        uq.objective_training(agent=agent, objective=objective, close_margin=3e-1)
    else:
        # Normal training.
        uq.train(agent=agent)

    # Get point c from agent's test.
    c = uq.testing(agent=agent)

    return c


def calc_frontier_scalarized(p, q, problem, solutions_known=None) -> list:
    """
    Return a list of supported solutions costs, this method is only valid to two objectives problems.
    Applies a dichotomous search to find all supported solutions costs.

    :param solutions_known: If we know the possible solutions, we can indicate them to the algorithm to improve the
        training of the agent. If is None, then is ignored.
    :param p: 2D point
    :param q: 2D point
    :param problem: A problem (agent)
    :return:
    """

    # A new list with p and q
    result = [p, q]

    # Create a new stack
    accumulate = list()

    # Push a vector with p and q in the stack
    accumulate.append(tuple(result))

    while len(accumulate) > 0:
        # Pop the next pair of points from the stack.
        a, b = accumulate.pop()

        # Order points nearest to the center using euclidean distance.
        a, b = tuple(um.order_points_by_center_nearest([a, b]))

        # Decompose points
        a_x, a_y = a
        b_x, b_y = b

        # Calculate the parameters of the new linear objective function (multiply by -1. to convert in maximize problem)
        w1 = np.multiply(a_y - b_y, -1.)
        w2 = np.multiply(b_x - a_x, -1.)

        # Solve P to find a new solution ang get its cost vector c.
        c = optimize(problem, w1, w2, solutions_known=solutions_known)

        # Decompose c vector.
        c_x, c_y = c

        if (w1 * a_x + w2 * a_y) != (w1 * c_x + w2 * c_y):
            # c is the cost of a new supported solution

            # Push new pair in the stack
            accumulate.append((a, c))

            # Push new pair in the stack
            accumulate.append((c, b))

            # Add c to the result
            result.append(c)

    return result
