from collections import Counter

from gym.spaces import Space


class Boolean(Space):

    def __init__(self):
        super().__init__()
        self.booleans = [True, False]

    def sample(self):
        return self.np_random.rand() < 0.5

    def contains(self, x):
        return isinstance(x, bool)

    def __repr__(self):
        return "Boolean({})".format(self.booleans)

    def __eq__(self, other):
        return isinstance(other, Boolean) and Counter(self.booleans) == Counter(other.booleans)
