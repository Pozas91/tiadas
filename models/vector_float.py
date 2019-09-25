"""
This class represent a vector with some features necessaries for our program.
This class have a vector of floats (float64).
"""

import numpy as np

from .dominance import Dominance
from .vector import Vector


class VectorFloat(Vector):
    """
    Class Vector with functions to work with float vectors.
    """

    def __init__(self, components):
        """
        Vector's init
        :param components:
        """
        super().__init__(components)

    def __ge__(self, other):
        """
        A vector is greater or equal than other when all components are greater or equal (or close) one by one.

        :param other:
        :return:
        """
        return np.all([
            a > b or np.isclose(a, b, atol=self.absolute_tolerance, rtol=self.relative_tolerance) for a, b in
            zip(self.components, other.components)
        ])

    def __gt__(self, other):
        """
        A vector is greater than other when all components are greater one by one.

        :param other:
        :return:
        """

        return np.all([
            a > b and not np.isclose(a, b, atol=self.absolute_tolerance, rtol=self.relative_tolerance) for a, b in
            zip(self.components, other.components)
        ])

    def __lt__(self, other):
        """
        A vector is less than other when all components are less one by one.

        :param other:
        :return:
        """
        return np.all([
            a < b and not np.isclose(a, b, atol=self.absolute_tolerance, rtol=self.relative_tolerance) for a, b in
            zip(self.components, other.components)
        ])

    def __le__(self, other):
        """
        A vector is less or equal than other when all components are less or equal (or close) one by one.

        :param other:
        :return:
        """
        return np.all([
            a < b or np.isclose(a, b, atol=self.absolute_tolerance, rtol=self.relative_tolerance) for a, b in
            zip(self.components, other.components)
        ])

    @staticmethod
    def m3_max_2_lists_not_duplicates(vectors: list) -> (list, list):
        """
        :param vectors: list of Vector objects.

        :return: a list with non-dominated vectors applying the m3 algorithm of
        Bentley, Clarkson and Levine (1990).
            We assume that:
                - We attempt to MAXIMIZE the value of each vector element.

            If, after all, two equal vectors are found, the algorithm keeps
            just one in the output lists.

            Return a tuple with a first list with the non-dominated list, and
            a second list with the dominated list.
        """

        non_dominated = list()
        dominated = list()
        vector_j = None

        for idx_i, vector_i in enumerate(vectors):

            # If is close to another vector, don't process this vector.
            if any([vector_i.all_close(v2=vector_to_examine) for vector_to_examine in
                    non_dominated + dominated]):
                continue

            # Prepare variables
            discarded = False
            idx_j = 0

            # While has more elements
            while idx_j < len(non_dominated) and not discarded:

                # Get vector and index
                vector_j = non_dominated[idx_j]

                # Get dominance
                dominance = vector_i.dominance(v2=vector_j)

                # `vector_i` dominate `vector_j`
                if dominance == Dominance.dominate:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                    # Add dominated vector_j
                    dominated.append(vector_j)

                # `vector_j` dominate `vector_i`
                elif dominance == Dominance.is_dominated:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                    # Set discarded to True
                    discarded = True

                    # Add dominated vector_i
                    dominated.append(vector_i)

                # If dominance is otherwise, continue searching
                if dominance == Dominance.otherwise:
                    # Search in next element
                    idx_j += 1
                else:
                    # Begin again
                    idx_j = 0

            if discarded:
                # Add vector at first
                non_dominated.insert(0, vector_j)
            else:
                # Add vector at end
                non_dominated.append(vector_i)

        return non_dominated, dominated
