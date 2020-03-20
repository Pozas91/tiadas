"""
Simple space to Boolean action spaces.
"""
from collections import Counter
from copy import copy

from gym.spaces import Space


class Boolean(Space):

    def __init__(self):
        super().__init__()
        self.booleans = [True, False]

    def sample(self):
        """
        Return a random boolean
        :return:
        """
        return self.np_random.rand() < 0.5

    def contains(self, x):
        """
        Check if `x` is a boolean.
        :param x:
        :return:
        """
        return isinstance(x, bool)

    def __repr__(self):
        """
        Return a string that represent a boolean space.
        :return:
        """
        return "Boolean({})".format(self.booleans)

    def __str__(self):
        """
        Convert boolean space to a string.
        :return:
        """
        return "Boolean({})".format(self.booleans)

    def __eq__(self, other):
        """
        Check if two boolean space are equals.
        :param other:
        :return:
        """
        return isinstance(other, Boolean) and Counter(self.booleans) == Counter(other.booleans)

    def __copy__(self):
        """
        Copy this space.
        :return:
        """
        return Boolean()

    def copy(self):
        """
        Method to copy this space.
        :return:
        """
        return copy(self)
