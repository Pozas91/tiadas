"""
Dominance Enum class, where indicate types of dominance that we are used in our code:
+ Dominance.dominate -> A vector dominate another vector.
+ Dominance.is_dominated -> A vector is dominated by another vector.
+ Dominance.equals -> Two vectors are equals or similar.
+ Dominance.otherwise -> Two vectors are independents.
"""
from enum import Enum


class Dominance(Enum):
    dominate = 1
    is_dominated = 2
    equals = 3
    otherwise = 4
