from collections import Counter

from gym.spaces import Space


class Bag(Space):

    def __init__(self, items: list):
        super().__init__()
        self.items = items
        self.n = len(items)
        self.index = 0

    def sample(self):
        return self.items[self.np_random.choice(len(self.items))]

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.n:
            x = self.items[self.index]
            self.index += 1
            return x
        else:
            raise StopIteration

    def contains(self, x: object):
        return x in self.items

    def __repr__(self):
        return "Bag({})".format(self.items)

    def __eq__(self, other):
        return isinstance(other, Bag) and Counter(self.items) == Counter(other.items)
