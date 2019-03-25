import copy

import math
import numpy as np

from models import Dominance


class Vector:
    """
    Class Vector with functions to work with vectors.
    """

    relative = 1e-5

    def __init__(self, components: list):
        """
        Vector's init
        :param components:
        """
        self.components = np.array(components)

    def __getitem__(self, item):
        """
        Get item from vector
        :param item:
        :return:
        """
        return self.components[item]

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
        return np.allclose(self, other, rtol=Vector.relative)

    def __str__(self):
        """
        Convert class to string
        :return:
        """
        return '[' + ' '.join([str(component) for component in self.components]) + ']'

    def __repr__(self):
        """
        String that represent class
        :return:
        """
        return '[' + ' '.join([str(component) for component in self.components]) + ']'

    def __hash__(self):
        """
        Hash to represent class
        :return:
        """
        return hash(repr(self))

    def __add__(self, other):
        """
        Sum two vectors
        :param other:
        :return:
        """

        if len(self) != len(other):
            raise ArithmeticError('Length of both vectors must be equal.')

        return Vector([x + y for x, y in zip(self, other)])

    def __sub__(self, other):
        """
        Sum two vectors
        :param other:
        :return:
        """

        if len(self) != len(other):
            raise ArithmeticError('Length of both vectors must be equal.')

        return Vector([x - y for x, y in zip(self, other)])

    def __ge__(self, other):
        """
        Self is greater or equal than other.
        Are greater or equal when all elements are greater or are close.
        :param other:
        :return:
        """

        return np.all([a > b or np.isclose(a, b, rtol=Vector.relative) for a, b in zip(self, other)])

    def __gt__(self, other):
        """
        self is greater than other.
        Are greater when are greater or equals and at least one element is greater.

        `SELF DOMINATE OTHER`
        :param other:
        :return:
        """
        return self >= other and self != other

    @property
    def magnitude(self):
        """
        Return magnitude of vector
        :return:
        """
        return math.sqrt(sum(component ** 2 for component in self))

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
            similar = np.allclose(v1, v2, rtol=Vector.relative)

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
        if Vector.similar(v1, v2):
            result = Dominance.equals

        # Check if `v1` is always greater or equal than `v2` (`v1` dominate `v2`), and at least one must be greater.
        elif v1 > v2:
            result = Dominance.dominate

        # Check if `v2` is always greater or equal than `v1` (`v1` is dominated by `v2`), and at least one must be
        # greater.
        elif v2 > v1:
            result = Dominance.is_dominated

        # Otherwise
        else:
            result = Dominance.otherwise

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

        non_dominated = list()
        vector_j = None

        for idx_i, vector_i in enumerate(vectors):

            discarded = False
            i = 0

            # While has more elements
            while i < len(non_dominated) and not discarded:

                # Get vector and index
                vector_j = non_dominated[i]
                idx_j = non_dominated.index(vector_j)

                dominance = Vector.dominance(v1=vector_i, v2=vector_j)

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

                    # If `vector_j` dominate `vector_i`
                    discarded = np.any(np.greater(vector_i, vector_j))

                # If dominance is otherwise, continue searching
                if dominance == Dominance.otherwise:
                    # Search in next element
                    i += 1
                else:
                    # Begin again
                    i = 0

            if discarded:
                # Add vector at first
                non_dominated.insert(0, vector_j)
            else:
                # Add vector at end
                non_dominated.append(vector_i)

        return non_dominated

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

        vector_j = None

        for idx_i, vector_i in enumerate(vectors):

            discarded = False
            i = 0

            # While has more elements
            while i < len(non_dominated) and not discarded:

                # Get vector and index
                vector_j = non_dominated[i]
                idx_j = non_dominated.index(vector_j)

                # Get dominance
                dominance = Vector.dominance(v1=vector_i, v2=vector_j)

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

                # `vector_i` and `vector_j` are similar or equals
                elif dominance == Dominance.equals:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                    # If `vector_j` dominate `vector_i`
                    discarded = np.any(np.greater(vector_i, vector_j))

                    # TODO: Esta parte no funciona del todo bien, habría que revisarlo.
                    # # If is discarded
                    # if discarded:
                    #     # Add dominated vector
                    #     dominated.append(vector_i)
                    # else:
                    #     # Add dominated vector
                    #     dominated.append(vector_j)

                # If dominance is otherwise, continue searching
                if dominance == Dominance.otherwise:
                    # Search in next element
                    i += 1
                else:
                    # Begin again
                    i = 0

            if discarded:
                # Add vector at first
                non_dominated.insert(0, vector_j)
            else:
                # Add vector at end
                non_dominated.append(vector_i)

        return non_dominated, dominated

    @staticmethod
    def m3_max_2_sets_with_repetitions(vectors: list):
        """

        :return: a list with non-dominated vectors applying m3 algorithm of Bentley, Clarkson and Levine (1990).
            We assume that:
                - Aren't two vector equals in list.
                - We attempt MAXIMIZE the value of all attributes.

            IMPORTANT:
                In this version, develop by AgenteMPQ5, set of non-dominated is formed by "buckets" which have all
                similar vectors. First of each bucket has restoTolong lowest value. That will be the one to keep the
                agent, while for all others you can choose to delete your predecessors (that is, they are updated from
                directly or indirectly).

            Return 2 sets, non-dominated set and dominated set.
        """

        non_dominated = list()
        dominated = list()

        for idx_i, vector_i in enumerate(vectors):

            discarded = False
            i = 0
            included_in_bucket = False
            bucket = None

            # While has more elements
            while i < len(non_dominated) and not discarded and not included_in_bucket:
                # Get vector and index
                bucket = non_dominated[i]

                # Get index of element
                idx_j = non_dominated.index(bucket)

                # Get first
                vector_j = bucket[0]

                # Vector dominance
                dominance = Vector.dominance(v1=vector_i, v2=vector_j)

                # `vector_i` dominate `vector_j`
                if dominance == Dominance.dominate:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                    # All vectors in bucket are added in dominated list
                    for vector_j in bucket:
                        # Add dominated vector
                        dominated.append(vector_j)

                # `vector_j` dominate `vector_i`
                elif dominance == Dominance.is_dominated:

                    # Remove non-dominated vector
                    non_dominated.pop(idx_j)

                    # Set discarded to True
                    discarded = True

                    # Add dominated vector
                    dominated.append(vector_j)

                # `vector_i` and `vector_j` are similar or equals
                elif dominance == Dominance.equals:

                    # Vector include in bucket
                    included_in_bucket = True

                    # If `vector_j` dominate `vector_i`
                    discarded = np.any(np.greater(vector_i, vector_j))

                    if discarded:
                        # Create a copy of vector_j
                        vector_j_copy = copy.deepcopy(vector_j)

                        # Add vector at first
                        bucket.insert(0, vector_j_copy)
                    else:
                        # Add vector at end
                        bucket.append(vector_i)

                # If dominance is otherwise, continue searching
                if dominance == Dominance.otherwise:
                    # Search in next element
                    i += 1
                else:
                    # Begin again
                    i = 0

            if discarded or included_in_bucket:
                # Add vector at first (bucket[:] is to pass value and not reference)
                non_dominated.insert(0, bucket[:])
            else:
                # List of vectors
                aux = [vector_i]

                # Add list at end
                non_dominated.append(aux)

        # Prepare list to non_dominated vectors
        non_dominated_unique = list()
        non_dominated_repeated = list()

        # for each bucket in non_dominated vector
        for bucket in non_dominated:
            # Get first (is unique)
            non_dominated_unique.append(bucket[0])

            # Get rest (are repeated)
            non_dominated_repeated += bucket[1:]

        return non_dominated_unique, dominated, non_dominated_repeated
