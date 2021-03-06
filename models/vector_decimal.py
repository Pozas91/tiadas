"""
This class represent a vector with some features necessaries for our program.
This class has a vector of decimals (decimal.Decimal).
"""

import numpy as np

import utils.numbers as un
from .dominance import Dominance
from .vector import Vector


class VectorDecimal(Vector):
    """
    Class Vector with functions to work with float vectors.
    """

    def __init__(self, components):
        """
        Vector'state init
        :param components: a numpy array, list or tuple of dtype items
        """

        super().__init__(np.array(components, dtype=float))

    def __ge__(self, other):
        """
        A vector is greater or equal than other when all components are greater or equal (or close) one by one.

        :param other:
        :return:
        """
        return all(
            a > b or un.are_equal_two_decimal_numbers(a=a, b=b) for a, b in zip(self.components, other.components)
        )

    def __gt__(self, other):
        """
        A vector is greater than other when all components are greater one by one.

        :param other:
        :return:
        """

        return all(
            a > b and not un.are_equal_two_decimal_numbers(a=a, b=b) for a, b in zip(self.components, other.components)
        )

    def __lt__(self, other):
        """
        A vector is less than other when all components are less one by one.

        :param other:
        :return:
        """
        return all(
            a < b and not un.are_equal_two_decimal_numbers(a=a, b=b) for a, b in zip(self.components, other.components)
        )

    def __le__(self, other):
        """
        A vector is less or equal than other when all components are less or equal (or close) one by one.

        :param other:
        :return:
        """
        return all(
            a < b or un.are_equal_two_decimal_numbers(a=a, b=b) for a, b in zip(self.components, other.components)
        )

    def __round__(self, n=None):
        """
        Return a vector with components rounded
        :param n:
        :return:
        """
        return self.__class__([
            un.round_with_precision(component, Vector.decimal_precision) for component in self.components
        ])

    def all_close(self, v2) -> bool:

        return all(
            un.are_equal_two_decimal_numbers(a=a, b=b) for a, b in zip(self.components, v2.components)
        )

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

    def dominance(self, v2) -> Dominance:
        """
        Check dominance between two Vector objects. Float values are allowed
        and treated with precision according to Vector.relative. It is assumed that all
        vector components are to be maximized.
        :param v2: a Vector object
        :return: an output value according to the Dominance enum.
        """

        v1_dominate = False
        v2_dominate = False

        for a, b in zip(self.components, v2.components):

            # Are equals or close...
            if un.are_equal_two_decimal_numbers(a=a, b=b):
                # Nothing to do at moment
                pass

            elif a > b:
                v1_dominate = True

                # If already dominate v2, then both vectors are independent.
                if v2_dominate:
                    return Dominance.otherwise

            # v1'state component_v1 is dominated by v2
            elif a < b:
                v2_dominate = True

                # If already dominate v1, then both vectors are independent.
                if v1_dominate:
                    return Dominance.otherwise

        if v1_dominate == v2_dominate:
            if v1_dominate:  # both, v1_dominate and v2_dominate are True -> vectors are indifferent
                return Dominance.otherwise

            # Are equals
            else:  # both, v1_dominate and v2_dominate are False -> vectors are (approximately) equal
                return Dominance.equals

        elif v1_dominate:  # v1 dominates v2
            return Dominance.dominate

        else:  # v2 dominates v1
            return Dominance.is_dominated
