from enum import Enum


class Dominance(Enum):
    dominate = 1
    is_dominated = 2
    equals = 3
    otherwise = 4
