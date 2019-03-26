"""
This class represent a vector with some features necessaries for our program.
This class have a vector, that could be
"""
import copy
import math

import numpy as np

from models import Dominance


class Vector:
    """
    Class Vector with functions to work with vectors.
    """

    # Relative margin to compare of similarity of two float elements.
    relative = 1e-5

    def __init__(self, components):
        """
        Vector's init
        :param components:
        """

        assert isinstance(components, (np.ndarray, list, set))
        self.components = np.array(components)

    def __getitem__(self, item):
        """
        Get item from vector:

        v1 = Vector([10, 2, 3])
        print(v1[0]) -> "10"
        print(v1.components[0]) -> "10"

        :param item:
        :return:
        """
        return self.components[item]

    def __len__(self):
        """
        Return first component from shape of self.components
        :return:
        """
        return self.components.shape[0]

    def __eq__(self, other):
        """
        True if two arrays have the same shape and elements, False otherwise.
        :param other:
        :return:
        """
        return np.array_equal(self, other)

    def __str__(self):
        """
        Return a string representation of the data in an array.
        :return:
        """
        return np.array_str(self.components)

    def __repr__(self):
        """
        Return the string representation of an array.
        :return:
        """
        return np.array_repr(self.components)

    def __hash__(self):
        """
        Hash to represent class
        :return:
        """
        return hash(tuple(self.components))

    def __add__(self, other):
        """
        Sum two vectors
        :param other:
        :return:
        """

        if self.components.shape != other.components.shape:
            raise ArithmeticError('Shape of both vectors must be equal.')

        return Vector(self.components + other.components)

    def __sub__(self, other):
        """
        Subtract two vectors
        :param other:
        :return:
        """

        if self.components.shape != other.components.shape:
            raise ArithmeticError('Shape of both vectors must be equal.')

        return Vector(self.components - other.components)

    def __mul__(self, other):
        """
        Multiply a vector
        :param other:
        :return:
        """

        pass

    def __ge__(self, other):
        """
        A vector is greater or equal than other when all components are greater or equal (or close) one by one.

        :param other:
        :return:
        """

        return np.all([
            a > b or np.isclose(a, b, rtol=Vector.relative) for a, b in zip(self.components, other.components)
        ])

    def __gt__(self, other):
        """
        A vector is greater than other when all components are greater one by one.

        :param other:
        :return:
        """
        return np.all(np.greater(self.components, other.components))

    def __lt__(self, other):
        """
        A vector is less than other when all components are less one by one.

        :param other:
        :return:
        """
        return np.all(np.less(self.components, other.components))

    def __le__(self, other):
        """
        A vector is less or equal than other when all components are less or equal (or close) one by one.

        :param other:
        :return:
        """
        return np.all([
            a < b or np.isclose(a, b, rtol=Vector.relative) for a, b in zip(self.components, other.components)
        ])

    @property
    def magnitude(self):
        """
        Return magnitude of vector
        :return:
        """
        return math.sqrt(np.sum(self.components ** 2))

    @staticmethod
    def all_close(v1, v2):
        """
        Returns True if two arrays are element-wise equal within a tolerance.

        If either array contains one or more NaNs, False is returned.
        :param v1:
        :param v2:
        :return:
        """

        return np.allclose(v1, v2, rtol=Vector.relative)

    @staticmethod
    def dominance(v1, v2):
        """
        Check if our vector dominate another vector.
        :param v1:
        :param v2:
        :return:
        """

        v1_dominate = False
        v2_dominate = False

        for idx, component in enumerate(v1.components):

            # Are equals or close...
            if np.isclose(v1.components[idx], v2.components[idx], rtol=Vector.relative):
                # Nothing to do at moment
                pass

            # In this component dominates v1
            elif v1.components[idx] > v2.components[idx]:
                v1_dominate = True

                # If already dominate v2, then both vectors are independent.
                if v2_dominate:
                    return Dominance.otherwise

            # In this component dominates v2
            elif v1.components[idx] < v2.components[idx]:
                v2_dominate = True

                # If already dominate v1, then both vectors are independent.
                if v1_dominate:
                    return Dominance.otherwise

        if v1_dominate == v2_dominate:
            # If both dominate, then both vectors are independent.
            if v1_dominate:
                return Dominance.otherwise

            # Are equals
            else:
                return Dominance.equals

        # v1 dominate to v2
        elif v1_dominate:
            return Dominance.dominate

        # v2 dominate to v1
        else:
            return Dominance.is_dominated

    @staticmethod
    def m3_max(vectors: list):
        """
        :return: a list with non-dominated vectors applying m3 algorithm of Bentley, Clarkson and Levine (1990).
            We assume that:
                - Aren't two vector equals in list.
                - We attempt MAXIMIZE the value of all attributes.

            If two vectors are equals, keep last one and discard the new vector.
        """

        non_dominated = list()
        vector_j = None

        for idx_i, vector_i in enumerate(vectors):

            discarded = False
            equals = False
            idx_j = 0

            # While has more elements
            while idx_j < len(non_dominated) and not discarded:

                # Get vector and index
                vector_j = non_dominated[idx_j]

                # Get dominance
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
                    # Stop to search (keep last one)
                    equals = True
                    break

                # If dominance is otherwise, continue searching
                if dominance == Dominance.otherwise:
                    # Search in next element
                    idx_j += 1
                else:
                    # Begin again
                    idx_j = 0

            # If both vectors are equals, do nothing
            if equals:
                pass
            elif discarded:
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

            Return 2 sets, non-dominated set and dominated set.
        """

        non_dominated = list()
        dominated = list()
        vector_j = None

        for idx_i, vector_i in enumerate(vectors):

            discarded = False
            equals = False
            idx_j = 0

            # While has more elements
            while idx_j < len(non_dominated) and not discarded:

                # Get vector and index
                vector_j = non_dominated[idx_j]

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
                    # Stop to search (keep last one)
                    equals = True
                    break

                # If dominance is otherwise, continue searching
                if dominance == Dominance.otherwise:
                    # Search in next element
                    idx_j += 1
                else:
                    # Begin again
                    idx_j = 0

            # If both vectors are equals, do nothing
            if equals:
                pass
            elif discarded:
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
                    idx_j += 1
                else:
                    # Begin again
                    idx_j = 0

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
