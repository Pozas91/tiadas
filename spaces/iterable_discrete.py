from gym.spaces import Discrete


class IterableDiscrete(Discrete):

    def __init__(self, n):
        super().__init__(n)
        self.index = 0

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < self.n:
            x = self.index
            self.index += 1
            return x
        else:
            raise StopIteration

    def __repr__(self):
        return "IterableDiscrete(%d)" % self.n

    def __eq__(self, other):
        return isinstance(other, IterableDiscrete) and self.n == other.n
