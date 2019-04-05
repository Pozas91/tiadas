"""
Useful functions to calculate the pareto frontier.
"""
import numpy as np
import pygmo as pg

from utils import q_learning


def optimize(agent, w1: float, w2: float, solutions_known=None) -> (float, float):
    """
    Try to find an c point to add in pareto's frontier.
    :param agent:
    :param solutions_known: If we know the possible solutions, we can indicate them to the algorithm to improve the
        training of the agent. If is None, then is ignored.
    :param w1:
    :param w2:
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
        q_learning.objective_training(agent=agent, objective=objective, close_margin=3e-1)
    else:
        # Normal training
        q_learning.train(agent=agent)

    # Get point c from agent's test.
    c = q_learning.testing(agent=agent)

    return c


def calc_frontier(p: (float, float), q: (float, float), problem, solutions_known=None) -> list:
    """
    Return a list of supported solutions costs
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

        # Order points nearest by center first.
        a, b = tuple(q_learning.order_points_by_center_nearest([a, b]))

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


def hypervolume(vector, reference=None) -> float:
    """
    By default, the pygmo library is used for minimization problems.
    In our case, we need it to work for maximization problems.
    :param vector: List of points limits of hypervolume
    :param reference: Reference point to calc hypervolume
    :return: hypervolume area.
    """

    if not reference:
        # Get min of all axis, and subtract 1.
        reference = (np.min(vector, axis=0) - 1)

    # Multiply by -1, to convert maximize problem into minimize problem.
    reference = np.multiply(reference, -1)
    vector = np.multiply(vector, -1)

    return pg.hypervolume(vector).compute(reference)


def sum_a_vector_and_a_set_of_vectors(v, v_set):
    """
    Perfoms a vector-sum between a vector v and a set of vectors V.
    :param v:
    :param v_set:
    :return:
    """
    return [v + vector for vector in v_set]
