"""
Wrapper method to calc hypervolume using pygmo library.
"""
import numpy as np
import pygmo as pg


def calc_hypervolume(list_of_vectors, reference=None) -> float:
    """
    By default, the pygmo library is used for minimization problems.
    In our case, we need it to work for maximization problems.
    :param list_of_vectors: List of vectors limits of hypervolume
    :param reference: Reference vector to calc hypervolume
    :return: hypervolume area.
    """

    if reference is None:
        # Get min of all axis, and subtract 1.
        reference = (np.min(list_of_vectors, axis=0) - 1)

    # Multiply by -1, to convert maximize problem into minimize problem.
    reference = np.multiply(reference, -1)
    list_of_vectors = np.multiply(list_of_vectors, -1)

    return pg.hypervolume(list_of_vectors).compute(reference)
