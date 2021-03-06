"""
This class represent a vector of integers (int32) with some necessary features for our program.
"""

import math
import operator
from decimal import Decimal as D

import numpy as np
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cityblock
from scipy.spatial.qhull import QhullError

import utils.models as um
from .dominance import Dominance


class Vector:
    """
    Class Vector with functions to work with int vectors.
    Vectors are represented internally as one-dimensional numpy arrays of the indicated type (int by default).
    """

    # Set decimal precision, by default two decimals (0.01)
    decimal_precision = D('0.01')

    @staticmethod
    def set_decimal_precision(decimal_precision: float):
        """
        Set precision for work with decimals
        :param decimal_precision:
        :return:
        """
        Vector.decimal_precision = D(str(decimal_precision))

    def __init__(self, components):
        """
        Vector'state init
        :param components: a numpy array, list or tuple of dtype items
        """
        self.components = np.array(components)

    def __getitem__(self, index):
        """
        Get the component in the index position

        v1 = Vector([10, 2, 3])
        print(v1[0]) -> "10"
        print(v1.components[0]) -> "10"

        :param index:
        :return: the component in the index position
        """
        return self.components[index]

    def __setitem__(self, key, value):
        """
        Set value to key in vector:

        v1 = Vector([10, 2, 3])

        v1[1] = 9 -> Vector([10, 9, 3])
        v1.components[1] = 9 -> Vector([10, 9, 3])
        :param key:
        :param value:
        :return:
        """
        self.components[key] = value

    def __abs__(self):
        """
        Return absolute value of components
        :return:
        """
        return self.__class__(np.absolute(self.components))

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
        Return a string representation of the train_data in an array.
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
        This method has four options:
            - A vector of same length has been given, return a new Vector with the sum of each pair of components.
            - A vector of different length has been given, throws an exception.
            - A int, sum that int of each component.
            - A float, sum that number as float, and remove decimals.
        :param other:
        :return:
        """

        if isinstance(other, Vector):
            return self.__class__(self.components + other.components)
        else:
            return self.__class__(self.components + other)

    def __sub__(self, other):
        """
        This method performs four different operations, depending on the nature of 'other':
            - A vector of same length has been given, return a new Vector with the subtraction of each pair of components.
            - A vector of different length has been given, throws an exception.
            - A int, subtract that int of each component.
            - A float, subtract that number as float, and remove decimals.

        :param other:
        :return:
        """

        if isinstance(other, Vector):
            return self.__class__(self.components - other.components)
        else:
            return self.__class__(self.components - other)

    def __mul__(self, other):
        """
        This method performs four different operations, depending on the nature of 'other':
            - A vector of same length has been given, return a new Vector with the multiply of each pair of components.
            - A vector of different length has been given, throws an exception.
            - A int, multiply that int of each component.
            - A float, multiply that number as float, and remove decimals.
        :param other:
        :return:
        """

        if isinstance(other, Vector):
            return self.__class__(self.components * other.components)
        else:
            return self.__class__(self.components * other)

    def __truediv__(self, other):
        """
        This method performs four different operations, depending on the nature of 'other':
            - A vector of same length has been given, return a new Vector with the division of each pair of components.
            - A vector of different length has been given, throws an exception.
            - A int, divide that int of each component.
            - A float, divide that number as float, and remove decimals.
        :param other:
        :return:
        """

        if isinstance(other, Vector):
            return self.__class__(np.divide(self.components, other.components))
        else:
            return self.__class__(np.divide(self.components, other))

    def __pow__(self, power, modulo=None):
        """
        This method has four options:
            - A vector of same length has been given, return a new Vector with the power of each pair of components.
            - A vector of different length has been given, throws an exception.
            - A int, power that int of each component.
            - A float, remove decimals, and power that number as int.
        :param power:
        :param modulo:
        :return:
        """

        return self.__class__(np.power(self.components, power))

    def __ge__(self, other):
        """
        A vector is greater or equal than other when all components are greater or equal one by one.
        - Throws an exception if both vectors lengths are different.
        :param other:
        :return:
        """

        return np.all(np.greater_equal(self.components, other.components))

    def __gt__(self, other):
        """
        A vector is greater than other when all components are greater one by one.
        - Throws an exception if both vectors lengths are different.
        :param other:
        :return:
        """
        return np.all(np.greater(self.components, other.components))

    def __lt__(self, other):
        """
        A vector is less than other when all components are less one by one.
        - Throws an exception if both vectors lengths are different.
        :param other:
        :return:
        """
        return np.all(np.less(self.components, other.components))

    def __le__(self, other):
        """
        A vector is less or equal than other when all components are less or equal (or close) one by one.
        - Throws an exception if both vectors lengths are different.
        :param other:
        :return:
        """
        return np.all(np.less_equal(self.components, other.components))

    def __round__(self, n=None):
        """
        Return a vector with components rounded
        :param n:
        :return:
        """
        return self.__class__([round(component, n) for component in self.components])

    @um.lazy_property
    def zero_vector(self):
        """
        Return a zero vector of same type and len that this vector
        :return:
        """
        return self * 0

    def copy(self):
        """
        Return a copy of this vector
        :return:
        """
        return self.__class__(self.components)

    def tolist(self):
        """
        Return all components of this vector in a list
        :return:
        """
        return self.components.tolist()

    def magnitude(self) -> float:
        """
        Return magnitude of vector
        :return:
        """
        return math.sqrt(np.sum(self.components ** 2))

    def all_close(self, v2) -> bool:
        """
        Returns True if two arrays are element-wise equal within a tolerance.

        If either array contains one or more NaNs, False is returned.

        As this vector is integer, the tolerance is 0, so this method is like equal comparision.
        :param v2:
        :return:
        """
        return np.equal(self.components, v2.components)

    def manhattan_distance(self, v2: 'Vector') -> float:
        """
        Manhattan distance between two vectors
        :param v2:
        :return:
        """
        return cityblock(u=self.components, v=v2.components)

    def euclidean_distance(self, v2: 'Vector') -> float:
        """
        Euclidean distance between two vectors
        :param v2:
        :return:
        """
        return np.linalg.norm(self.components - v2.components)

    def distance_to_origin(self) -> float:
        """
        Distance between a vector and origin vector
        :return:
        """
        return self.euclidean_distance(v2=self.zero_vector)

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
            if a == b:
                # Nothing to do at moment
                pass

            elif a > b:
                v1_dominate = True

                # If already dominate v2, then both vectors are independent.
                if v2_dominate:
                    return Dominance.otherwise

            # v1'state component is dominated by v2
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

    @staticmethod
    def order_vectors_by_origin_nearest(vectors: list) -> list:
        """
        Order given vectors by nearest to origin
        :param vectors:
        :return:
        """

        # Get all vectors with its distance to origin.
        distances = {tuple(vector): Vector.distance_to_origin(vector) for vector in vectors}

        # Sort dictionary by value from lower to higher.
        return sorted(distances, key=distances.get, reverse=False)

    @staticmethod
    def convex_hull(vectors: set) -> list:
        """
        :param vectors: |vectors| >= 3
            if |vectors| < 3, then return the vectors
        :return:
        """

        # Convert to list to make it indexable
        vectors = list(vectors)

        if len(vectors) >= 3:

            try:
                hull = ConvexHull(vectors, incremental=False)
                vectors = list(operator.itemgetter(*hull.vertices)(vectors))
            except QhullError as e:
                # Expected error
                pass

        return vectors

    @staticmethod
    def m3_max(vectors: set) -> list:
        """
        :param vectors : set of Vector objects, float values are assumed.

        :return: a list with non-dominated vectors applying the m3 algorithm of
        Bentley, Clarkson and Levine (1990).
            We assume that:
                - There are not two equal vectors in the input list.
                - We attempt to MAXIMIZE the value of each vector element.

            If, after all, two equal vectors are found, the algorithm keeps
            just one in the output list.
        """

        non_dominated = list()
        vector_j = None

        for vector_i in vectors:

            discarded = False
            equals = False
            idx_j = 0

            # While has more elements
            while idx_j < len(non_dominated) and not discarded:

                # Get vector and index
                vector_j = non_dominated[idx_j]

                # Get dominance
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
                    # Stop to search (keep last one)
                    equals = True
                    break

                # If dominance is otherwise, continue searching
                if dominance is Dominance.otherwise:
                    # Search in next element
                    idx_j += 1
                else:
                    # Begin again
                    idx_j = 0

            # If both vectors are equal, do nothing
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
    def m3_max_2_lists(vectors: set) -> (list, list):
        """
        :param vectors : set of Vector objects.

        :return: a list with non-dominated vectors applying the m3 algorithm of
        Bentley, Clarkson and Levine (1990).
            We assume that:
                - There are not two equal vectors in the input list.
                - We attempt to MAXIMIZE the value of each vector element.

            If, after all, two equal vectors are found, the algorithm keeps
            just one in the non_dominate list, but is duplicate in dominated list.

            Return a tuple with a first list with the non-dominated list, and
            a second list with the dominated list.
        """

        non_dominated = list()
        dominated = list()
        vector_j = None

        for vector_i in vectors:

            discarded = False
            equals = False
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
    def m3_max_2_lists_not_duplicates(vectors: set) -> (list, list):
        """
        :param vectors: set of Vector objects.

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

        for vector_i in vectors:

            # If vector_i is in non_dominated or dominated, not process it.
            if vector_i in non_dominated + dominated:
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

    @staticmethod
    def m3_max_2_lists_with_buckets(vectors: set) -> (list, list):
        """
        :param vectors: set of Vector objects.

        :return: a list with non-dominated vectors applying the m3 algorithm of
        Bentley, Clarkson and Levine (1990).
            We assume that:
                - We attempt to MAXIMIZE the value of each vector element.

            Return a tuple with a first list with the non-dominated bucket list, and a
            second list with the dominated list..
        """

        non_dominated = list()
        dominated = list()

        for vector_i in vectors:

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
                    dominated.append(vector_i)

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

        return non_dominated, dominated

    @staticmethod
    def m3_max_2_lists_with_repetitions(vectors: set) -> (list, list, list):
        """
        :param vectors: set of Vector objects.

        :return: a list with non-dominated vectors applying the m3 algorithm of
        Bentley, Clarkson and Levine (1990).
            We assume that:
                - We attempt to MAXIMIZE the value of each vector element.

            Return a tuple with a first list with the non-dominated unique list, a
            second list with the dominated list, and a third list with non-dominated duplicate list.
        """

        non_dominated, dominated = Vector.m3_max_2_lists_with_buckets(vectors=vectors)

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
