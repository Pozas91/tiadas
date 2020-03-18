"""
Wrapper method to calc hypervolume using pygmo library.

EXAMPLE OF USE:

    WITHOUT REFERENCE:

        # Prepare list of vectors
        vectors = [Vector([0, 0]), Vector([-3, 1]), Vector([1, 1])]

        # Calc hypervolume of those vectors
        calc_hypervolume(vectors=vectors)   # Returns 10.0

    WITH REFERENCE:

        # Prepare list of vectors
        vectors = [Vector([0, 0]), Vector([-3, 1]), Vector([1, 1])]

        # Calc hypervolume of those vectors
        calc_hypervolume(vectors=vectors, reference=Vector([-4, -1]))   # Returns 10.0

        calc_hypervolume(vectors=vectors, reference=Vector([-6, -6]))   # Returns 49.0

"""
import numpy as np
import pygmo as pg

from models import Vector


def calc_hypervolume(vectors: list, reference: Vector) -> float:
    """
    By default, the pygmo library is used for minimization problems.
    In our case, we need it to work for maximization problems.
    :param vectors: List of vectors limits of hypervolume
    :param reference: Reference vector to calc hypervolume
    :return: hypervolume area.
    """
    # Multiply by -1, to convert maximize problem into minimize problem.
    return pg.hypervolume([v.components * -1 for v in vectors]).compute(reference.components * -1)
