"""
Defines a Bag space, which has a list of items. Is specially util for environments with dynamic action spaces.
"""
from collections import Counter
from copy import copy

from gym.spaces import Space


class Bag(Space):

    def __init__(self, items: list):
        """

        :param items: list of items to save in the space.
        """
        super().__init__()
        self.items = items
        self.n = len(items)
        self.index = 0

    def sample(self):
        """
        Return a random item
        :return:
        """
        return self.items[self.np_random.choice(len(self.items))]

    def __iter__(self):
        """
        Iter on bag space.
        :return:
        """
        self.index = 0
        return self

    def __next__(self):
        """
        Return a next element of bag
        :return:
        """
        if self.index < self.n:
            x = self.items[self.index]
            self.index += 1
            return x
        else:
            raise StopIteration

    def contains(self, x: object):
        """
        Check if `x` is in bag.
        :param x:
        :return:
        """
        return x in self.items

    def __repr__(self):
        """
        Return a string that represent this action space.
        :return:
        """
        return "Bag({})".format(self.items)

    def __str__(self):
        """
        Convert this class to a string
        :return:
        """
        return "Bag({})".format(self.items)

    def __eq__(self, other):
        """
        Check if two bags are equals
        :param other:
        :return:
        """
        return isinstance(other, Bag) and Counter(self.items) == Counter(other.items)

    def __copy__(self):
        """
        Return a copy of this bag
        :return:
        """
        return Bag(self.items)

    def copy(self):
        """
        Method to copy this bag
        :return:
        """
        return copy(self)
