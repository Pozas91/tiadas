from collections import Counter

from gym.spaces import Space


class DynamicSpace(Space):

    def __init__(self, items: list):
        super().__init__()
        self.items = items
        self.n = len(items)

    def sample(self):
        return self.np_random.choice(self.items)

    def contains(self, x: object):
        return x in self.items

    def __getitem__(self, item: int) -> object:
        return self.items[item]

    def __repr__(self):
        return "DynamicSpace({})".format(self.items)

    def __eq__(self, other):
        return isinstance(other, DynamicSpace) and Counter(self.items) == Counter(other.items)
