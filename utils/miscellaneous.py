"""
Useful functions used in this project
"""
import re

from configurations import yml_indent
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


def sum_a_vector_and_a_list_of_vectors(v: Vector, v_list: list):
    """
    Performs a vector-sum between a vector v and a set of vectors V.
    :param v:
    :param v_list:
    :return:
    """
    return [v + vector for vector in v_list]


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
            result += ' ' * (level * yml_indent) + "{}:\n".format(k) + structures_to_yaml(data=v, level=level + 1)
        else:
            result += ' ' * (level * yml_indent) + "{}: {}\n".format(k, v)

    return result
