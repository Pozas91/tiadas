"""
Dominance Enum class, indicates the result of a dominance check of one vector v with another v2:
+ Dominance.dominate -> v dominates v2
+ Dominance.is_dominated -> v is dominated by v2
+ Dominance.equals -> Both vectors are equals or similar.
+ Dominance.otherwise -> vectors are indifferent to each other (they are not equal or similar an no one dominates
                         the other.
"""
from enum import Enum


class Dominance(Enum):
    dominate = 1
    is_dominated = 2
    equals = 3
    otherwise = 4
