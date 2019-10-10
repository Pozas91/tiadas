from gym.spaces import Space


class Boolean(Space):

    def sample(self):
        return self.np_random.rand() < 0.5

    def contains(self, x):
        return isinstance(x, bool)

    def __init__(self):
        super().__init__()
        self.booleans = [True, False]
