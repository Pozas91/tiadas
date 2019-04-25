"""
Useful functions used in this project
"""
import numpy as np


def lists_to_tuples(x):
    """
    Convert a list given into tuple recursively.
    :param x:
    :return:
    """

    if not isinstance(x, list):
        return x

    return tuple(map(lists_to_tuples, x))


def sum_a_vector_and_a_set_of_vectors(v, v_set):
    """
    Performs a vector-sum between a vector v and a set of vectors V.
    :param v:
    :param v_set:
    :return:
    """
    return [v + vector for vector in v_set]


def is_close(a: float, b: float, relative=1e-3):
    """
    Check if two float numbers are close
    :param a:
    :param b:
    :param relative:
    :return:
    """
    return abs(a - b) <= relative


def distance_between_two_points(a, b) -> float:
    """
    Simple distance between two points
    :param a:
    :param b:
    :return:
    """
    return np.linalg.norm(np.array(a) - np.array(b))


def distance_to_origin(a) -> float:
    """
    Distance between a point and origin
    :param a:
    :return:
    """

    a = np.array(a)
    b = np.zeros_like(a)

    return distance_between_two_points(a=a, b=b)


def order_points_by_center_nearest(points: list) -> list:
    """
    Order given points by center nearest
    :param points:
    :return:
    """

    # Get all points with its distance to origin.
    distances = {point: distance_to_origin(point) for point in points}

    # Sort dictionary by value from lower to higher.
    return sorted(distances, key=distances.get, reverse=False)


def weighted_sum(rewards, weights) -> float:
    """
    Simple Weighted-Sum function
    :param rewards:
    :param weights:
    :return:
    """
    return float(np.sum(np.multiply(rewards, weights)))
