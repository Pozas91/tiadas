"""
Define a iterable integer. From zero to `n`
"""
from copy import copy

from gym.spaces import Discrete


class IterableDiscrete(Discrete):

    def __init__(self, n):
        """
        :param n: Upper limit to iterate
        """
        super().__init__(n)
        self.index = 0

    def __iter__(self):
        """
        Provides iteration of the class.
        :return:
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Return next element of the space
        :return:
        """
        if self.index < self.n:
            x = self.index
            self.index += 1
            return x
        else:
            raise StopIteration

    def __repr__(self):
        """
        Return a string that represent the iterable integer. p.e. `IterableDiscrete(2)`
        :return:
        """
        return "IterableDiscrete({})".format(self.n)

    def __str__(self):
        """
        Convert this class to a string to return it.
        :return:
        """
        return "IterableDiscrete({})".format(self.n)

    def __eq__(self, other):
        """
        Check if two iterable discrete space are equals
        :param other:
        :return:
        """
        return isinstance(other, IterableDiscrete) and self.n == other.n

    def __copy__(self):
        """
        Return a copy of this space
        :return:
        """
        return IterableDiscrete(self.n)

    def copy(self):
        """
        Method to copy this state.
        :return:
        """
        return copy(self)
