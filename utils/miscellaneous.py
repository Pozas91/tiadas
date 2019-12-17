"""
Useful functions used in this project
"""
import math
import re

import numpy as np
from scipy.spatial.distance import cityblock

from models import Vector


def lists_to_tuples(x):
    """
    Convert a list given into tuple recursively.
    :param x:
    :return:
    """

    if not isinstance(x, list):
        return x

    return tuple(map(lists_to_tuples, x))


def tuples_to_string(x, level: int = 1):
    """
    Convert nested tuples into a string.
    :param x:
    :param level:
    :return:
    """

    if not isinstance(x, tuple):
        return str(x)

    separator = ':' * level

    return separator.join(map(lambda y: tuples_to_string(y, level + 1), x))


def string_to_tuples(x, level: int = 1):
    """
    Invert operation of tuples_to_string
    :param x:
    :param level:
    :return:
    """

    pass


def sum_a_vector_and_a_list_of_vectors(v: Vector, v_list: list):
    """
    Performs a vector-sum between a vector v and a set of vectors V.
    :param v:
    :param v_list:
    :return:
    """
    return [v + vector for vector in v_list]


def euclidean_distance(a: Vector, b: Vector) -> float:
    """
    Euclidean distance between two vectors
    :param a:
    :param b:
    :return:
    """
    return np.linalg.norm(a - b)


def distance_to_origin(a: Vector) -> float:
    """
    Distance between a vector and origin vector
    :param a:
    :return:
    """
    return euclidean_distance(a=a, b=a.zero_vector)


def order_vectors_by_origin_nearest(vectors: list) -> list:
    """
    Order given vectors by nearest to origin
    :param vectors:
    :return:
    """

    # Get all vectors with its distance to origin.
    distances = {tuple(vector): distance_to_origin(vector) for vector in vectors}

    # Sort dictionary by value from lower to higher.
    return sorted(distances, key=distances.get, reverse=False)


def manhattan_distance(a: Vector, b: Vector) -> float:
    """
    Manhattan distance between two vectors
    :param a: Vector a
    :param b: Vector b
    :return:
    """

    return cityblock(u=a.components, v=b.components)


def math_is_close(a: float, b: float, rel_tol: float = 1e-09, abs_tol: float = 0.0):
    """

    :param a:
    :param b:
    :param rel_tol:
    :param abs_tol:
    :return:
    """

    # return round(abs(a - b), Vector.decimal_allowed) <= max(
    #     round(rel_tol * max(abs(a), abs(b)), Vector.decimal_allowed), abs_tol)
    return math.isclose(a, b, rel_tol, abs_tol)


def str_to_snake_case(text: str) -> str:
    """
    Convert a text given to snake case text.
    This is used, for example, to convert environment names and
    write them in files.
    :param text:
    :return:
    """
    s1 = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string=text)
    return re.sub('([a-z0-9])([A-Z])', r'\1_\2', s1).lower()


def structures_to_yaml(data, level: int = 0) -> str:
    result = ''

    for k, v in sorted(data.items(), key=lambda x: x[0]):
        if isinstance(v, dict):
            result += ' ' * (level * 2) + "{}:\n".format(k) + structures_to_yaml(data=v, level=level + 1)
        else:
            result += ' ' * (level * 2) + "{}: {}\n".format(k, v)

    return result
