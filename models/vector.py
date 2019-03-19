import copy
from enum import Enum

import numpy as np


class DOMINANCE(Enum):
    dominate = 1
    is_dominated = 2
    equals = 3
    otherwise = 4


class Vector:
    """
    Class Vector with functions to work with vectors.
    """

    def __init__(self, components: list):
        """
        Vector's init
        :param components:
        """
        self.components = np.array(components)

    def __len__(self):
        """
        Override len functions
        :return:
        """
        return len(self.components)

    def __eq__(self, other):
        """
        Check if two vectors are equals
        :param other:
        :return:
        """
        return np.array_equal(self.components, other.components)

    def __str__(self):
        """
        String that represent class
        :return:
        """
        return self.components

    @staticmethod
    def similar(v1, v2):
        """
        Check if two vectors are similar
        :param v1:
        :param v2:
        :return:
        """

        # Get `v1` length
        length = len(v1)

        # First, must be same length
        similar = length == len(v2)

        # If both have same length
        if similar:
            # Check if all values are closed.
            similar = np.allclose(v1.components, v2.components)

        return similar

    @staticmethod
    def dominance(v1, v2):
        """
        Check if our vector dominate another vector.
        :param v1:
        :param v2:
        :return:
        """

        # Check if are equals
        if v1 == v2 or Vector.similar(v1, v2):
            result = DOMINANCE.equals

        # Check if `v1` is always greater than `v2` (`v1` dominate `v2`)
        elif np.all(np.greater(v1.components, v2.components)):
            result = DOMINANCE.dominate

        # Check if `v2` is always greater than `v1` (`v1` is dominated by `v2`)
        elif np.all(np.greater(v2.components, v1.components)):
            result = DOMINANCE.is_dominated

        # Otherwise
        else:
            result = DOMINANCE.otherwise

        return result

    @staticmethod
    def m3_max(vectors: list):
        """

        :return: a list with non-dominated vectors applying m3 algorithm of Bentley, Clarkson and Levine (1990).
            We assume that:
                - Aren't two vector equals in list.
                - We attempt MAXIMIZE the value of all attributes.

            IMPORTANT: Between V equals vectors only the lowest value is chosen
                (which in the case of the objects Vj will be the index j).
        """

        result = list()

        for idx_i, vector_i in enumerate(vectors):

            discarded = False

            for idx_j, vector_j in enumerate(result):

                dominance = Vector.dominance(v1=vector_i, v2=vector_j)

                # `vector_i` dominate `vector_j`
                if dominance == DOMINANCE.dominate:
                    result.pop(idx_j)

                # `vector_j` dominate `vector_i`
                elif dominance == DOMINANCE.is_dominated:
                    result.pop(idx_j)

                    discarded = True

                # `vector_i` and `vector_j` are similar or equals
                elif dominance == DOMINANCE.equals:
                    # TO DO
                    pass

                # If a vector is discarded, stop for loop.
                if discarded:
                    break

            if discarded:
                # Create a copy of vector_j
                vector_j_copy = copy.deepcopy(vector_j)

                # Add vector at first
                result.insert(0, vector_j_copy)
            else:
                # Add vector at end
                result.append(vector_i)

        return result

    @staticmethod
    def m3_max_2_sets(vectors: list):
        """

        :return: a list with non-dominated vectors applying m3 algorithm of Bentley, Clarkson and Levine (1990).
            We assume that:
                - Aren't two vector equals in list.
                - We attempt MAXIMIZE the value of all attributes.

            IMPORTANT: Between V equals vectors only the lowest value is chosen
                (which in the case of the objects Vj will be the index j).

            Return 2 sets, non-dominated set and dominated set.
        """

        non_dominated = list()
        dominated = list()

        for idx_i, vector_i in enumerate(vectors):

            discarded = False

            for idx_j, vector_j in enumerate(non_dominated):

                dominance = Vector.dominance(v1=vector_i, v2=vector_j)

                # `vector_i` dominate `vector_j`
                if dominance == DOMINANCE.dominate:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                    # Add dominated vector
                    dominated.append(vector_j)

                # `vector_j` dominate `vector_i`
                elif dominance == DOMINANCE.is_dominated:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                    # Set discarded to True
                    discarded = True

                    # Add dominated vector
                    dominated.append(vector_j)

                # `vector_i` and `vector_j` are similar or equals
                elif dominance == DOMINANCE.equals:
                    # TO DO
                    pass

                # If a vector is discarded, stop for loop.
                if discarded:
                    break

            if discarded:
                # Create a copy of vector_j
                vector_j_copy = copy.deepcopy(vector_j)

                # Add vector at first
                non_dominated.insert(0, vector_j_copy)
            else:
                # Add vector at end
                non_dominated.append(vector_i)

        return non_dominated
