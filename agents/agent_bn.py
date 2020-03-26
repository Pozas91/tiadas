"""
Agent NB, like Agent W, but using convex hull instead non dominates.

We consider the following operations and functions:
• CH(X), the set of convex-hull vectors from vector set X ⊂ R^n.
• r(state, a, state'), the vector reward associated to transition (state, a, state').
• p(state, a, state'), the transition probability associated to transition (state, a, state').
"""

from models import Vector
from . import AgentW


class AgentBN(AgentW):

    @staticmethod
    def filter_vectors(vectors: set) -> list:
        # CH[vectors]
        return Vector.convex_hull(vectors=vectors)
