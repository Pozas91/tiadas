"""
Such as Vector class, but has information about the action associate with this vector.
"""
import numpy as np

from .dominance import Dominance
from .vector import Vector


class ActionVector:
    """
    Class ActionVector with functions to work with int vectors.
    """

    def __init__(self, action: int, vector: Vector):
        """

        :param action: Action to associate a vector.
        :param vector: Vector instance, could be int or float vector.
        """

        # Action associate
        self.action = action

        # Vector associate
        self.vector = vector

    def __str__(self):
        """
        Return a string representation of the data in an array, with action associate.
        :return:
        """
        return 'A: {}, V: {}'.format(self.action, np.array_str(self.vector.components))

    def __repr__(self):
        """
        Return the string representation of an array, with action associate.
        :return:
        """
        return 'A: {}, V: {}'.format(self.action, np.array_repr(self.vector.components))

    def dominance(self, v2) -> Dominance:
        """
        Check dominance between two ActionVector objects.
        :param v2:
        :return:
        """
        return self.vector.dominance(v2.vector)

    @staticmethod
    def actions_occurrences_based_m3_with_repetitions(vectors: list, number_of_actions: int) -> list:
        """
        :param number_of_actions:
        :param vectors: list of Vector objects.

        :return: Return a list of non_dominated vectors occurrences per action. Applying the m3 algorithm of
        Bentley, Clarkson and Levine (1990).
            We assume that:
                - We attempt to MAXIMIZE the value of each vector element.
        """

        non_dominated = list()
        actions = [0] * number_of_actions

        for idx_i, vector_i in enumerate(vectors):

            discarded = False
            idx_j = 0
            included_in_bucket = False
            bucket = None

            # While has more elements
            while idx_j < len(non_dominated) and not discarded and not included_in_bucket:

                # Get vector and index
                bucket = non_dominated[idx_j]

                # Get first
                vector_j = bucket[0]

                # Vector dominance
                dominance = vector_i.dominance(v2=vector_j)

                # `vector_i` dominate `vector_j`
                if dominance == Dominance.dominate:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                # `vector_j` dominate `vector_i`
                elif dominance == Dominance.is_dominated:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                    # Set discarded to True
                    discarded = True

                # `vector_i` and `vector_j` are similar or equals
                elif dominance == Dominance.equals:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                    # Vector include in bucket
                    included_in_bucket = True

                    # Add vector_i to exists bucket.
                    bucket.append(vector_i)

                # If dominance is otherwise, continue searching
                if dominance == Dominance.otherwise:
                    # Search in next element
                    idx_j += 1
                else:
                    # Begin again
                    idx_j = 0

            if discarded or included_in_bucket:
                # Add all bucket at first of non_dominated list (bucket[:] is to pass value and not reference)
                non_dominated.insert(0, bucket[:])
            else:
                # List of vectors
                aux = [vector_i]

                # Add list at end
                non_dominated.append(aux)

        # for each bucket in non_dominated
        for bucket in non_dominated:
            # for each vector in bucket
            for vector in bucket:
                # Increment vector action
                actions[vector.action] += 1

        return actions
