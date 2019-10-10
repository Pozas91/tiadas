"""
Class that represents a vector with an associated integer index, that can be used, for example, as an identifier.
"""
import numpy as np

from .dominance import Dominance
from .vector import Vector


class IndexVector:
    """
    Class IndexVector with functions to work with int vectors.
    """

    def __init__(self, index: int, vector: Vector):
        """
        :param index: Index to associate a vector.
        :param vector: Vector instance, could be int or float vector.
        """

        # Action associate
        self.index = index

        # Vector associate
        self.vector = vector

    def __str__(self):
        """
        Return a string representation of the data in an array, with index associate.
        :return:
        """
        return 'I: {}, V: {}'.format(self.index, np.array_str(self.vector.components))

    def __mul__(self, other):
        """
        Multiply a index vector by other.
        :param other:
        :return:
        """
        return IndexVector(index=self.index, vector=self.vector * other)

    def __repr__(self):
        """
        Return the string representation of an array, with index associate.
        :return:
        """
        return '{} - {}'.format(self.index, self.vector.__repr__())

    def dominance(self, v2) -> Dominance:
        """
        Check dominance between two IndexVector objects.
        :param v2:
        :return:
        """
        return self.vector.dominance(v2.vector)

    @staticmethod
    def actions_occurrences_based_m3_with_repetitions(vectors: list, actions: list,
                                                      returns_vectors: bool = False) -> dict:
        """
        This function receives a list on IndexVector objects (where de index of each vector is an integer representing
        an action), and a list of indices (each one representing an action used as index in the previous list).

        The function calculates the set of non-dominated vectors in the first list (with repetitions), and counts,
        for each action, the number of associated non-dominated vectors.

        Then, a dictionary is returned, where keys are actions, and values are non-dominated vector counts.

        :param returns_vectors:
        :param actions:
        :param vectors: list of Vector objects.

        :return: Return a dictionary with each action given and occurrences of that action applying the m3 algorithm of
        Bentley, Clarkson and Levine (1990).
            We assume that:
                - We attempt to MAXIMIZE the value of each vector element.

        If return_vectors is True, return a dictionary with the vectors for each action.
        """

        non_dominated = list()

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
                if dominance is Dominance.dominate:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                # `vector_j` dominate `vector_i`
                elif dominance is Dominance.is_dominated:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                    # Set discarded to True
                    discarded = True

                # `vector_i` and `vector_j` are similar or equals
                elif dominance is Dominance.equals:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                    # Vector include in bucket
                    included_in_bucket = True

                    # Add vector_i to exists bucket.
                    bucket.append(vector_i)

                # If dominance is otherwise, continue searching
                if dominance is Dominance.otherwise:
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

        if returns_vectors:
            vectors_dict = {action: [] for action in actions}

            # for each bucket in non_dominated
            for bucket in non_dominated:
                # for each vector in bucket
                for vector in bucket:
                    vectors_dict[vector.index].append(vector.vector)

            return vectors_dict

        else:

            # Prepare actions dict
            actions_dict = {action: 0 for action in actions}

            # for each bucket in non_dominated
            for bucket in non_dominated:
                # for each vector in bucket
                for vector in bucket:
                    # Increment vector action
                    actions_dict[vector.index] += 1

            return actions_dict
