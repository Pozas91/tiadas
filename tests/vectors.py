"""
Unit tests file where testing vector model.
"""

import math
import random as rnd
import unittest
from copy import deepcopy

import numpy as np

from models import Vector, Dominance


class TestVectors(unittest.TestCase):
    first_quadrant = None
    second_quadrant = None
    third_quadrant = None
    fourth_quadrant = None

    def setUp(self):
        relative = Vector.relative

        self.first_quadrant = [
            Vector([0, 6]),
            Vector([1, 6]),
            Vector([2, 5]),
            Vector([2, 4]),
            Vector([2, 2]),
            Vector([3, 4]),
            Vector([4, 3]),
            Vector([4, 1]),
            Vector([5, 3]),
            Vector([5, 2]),
            Vector([6, 0]),

            # Repeats
            Vector([0, 6]),
            Vector([4, 1]),

            # Similar
            Vector([5 + relative, 3 - relative]),
            Vector([2 + relative, 4 - relative]),
        ]

        self.second_quadrant = [
            Vector([-1, 0]),
            Vector([-3, 4]),
            Vector([-4, 2]),
            Vector([-4, 7]),
            Vector([-6, 6]),
            Vector([-6, 0]),
            Vector([-8, 2]),

            # Repeats
            Vector([-1, 0]),
            Vector([-6, 6]),

            # Similar
            Vector([-2 + relative, -1 + relative]),
            Vector([-4 - relative, -7 + relative]),
        ]

        self.third_quadrant = [
            Vector([-1, -4]),
            Vector([-2, -1]),
            Vector([-3, -6]),
            Vector([-4, -2]),
            Vector([-5, -4]),
            Vector([-7, -1]),

            # Repeats
            Vector([-1, -4]),
            Vector([-7, -1]),

            # Similar
            Vector([-2 - relative, -1 - relative]),
            Vector([-4 + relative, -2 + relative]),
        ]

        self.fourth_quadrant = [
            Vector([2, -1]),
            Vector([3, -2]),
            Vector([1, -4]),
            Vector([3, -2]),
            Vector([3, -5]),
            Vector([5, -6]),
            Vector([7, -3]),
            Vector([10, -1]),

            # Repeats
            Vector([2, -1]),
            Vector([10, -1]),

            # Similar
            Vector([-7 + relative, -3 - relative]),
            Vector([10 + relative, -1 + relative]),
        ]

    def tearDown(self):
        pass

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        components = [rnd.uniform(-100., 100.) for _ in range(rnd.randint(2, 10))]

        # List
        vector = Vector(components)
        self.assertTrue(isinstance(vector.components, np.ndarray))

        # Set
        vector = Vector(set(components))
        self.assertTrue(isinstance(vector.components, np.ndarray))

        # ndarray
        vector = Vector(np.asarray(components))
        self.assertTrue(isinstance(vector.components, np.ndarray))

    def test_length(self):
        """
        Testing if override len() operator works!
        :return:
        """

        two_length = Vector([rnd.uniform(-100., 100.) for _ in range(2)])
        self.assertEqual(2, len(two_length))

        ################################################################################################################

        three_length = Vector([rnd.uniform(-100., 100.) for _ in range(3)])
        self.assertEqual(3, len(three_length))

        ################################################################################################################

        n = rnd.randint(1, 20)
        n_length = Vector([rnd.uniform(-100., 100.) for _ in range(n)])
        self.assertEqual(n, len(n_length))

    def test_equal(self):
        """
        Testing if override = operator works!
        :return:
        """

        x = Vector([rnd.uniform(-100., 100.) for _ in range(rnd.randint(2, 10))])
        y = deepcopy(x)

        self.assertEqual(y, x)

    def test_gte(self):
        """
        Testing if override >= operator works!
        :return:
        """

        x = Vector([5 + Vector.relative, 3 - Vector.relative])
        y = Vector([4, 3])
        z = Vector([5, 3])

        self.assertTrue(x >= y)
        self.assertTrue(z >= y)
        self.assertTrue(x >= z)
        self.assertTrue(z >= y)
        self.assertFalse(y >= x)

    def test_gt(self):
        """
        Testing if override > (dominate) operator works!
        :return:
        """

        x = Vector([5 + Vector.relative, 3 + Vector.relative])
        y = Vector([4, 3])
        z = Vector([5, 3])

        self.assertFalse(x > y)
        self.assertFalse(z > y)
        self.assertFalse(x > z)
        self.assertFalse(y > x)

    def test_str(self):
        """
        Testing if override str operator works!
        :return:
        """

        x = Vector([1, 2, 3])
        self.assertEqual(np.array_str(x.components), str(x))

        ################################################################################################################

        x = Vector([1, -2])
        self.assertEqual(np.array_str(x.components), str(x))

        ################################################################################################################

        x = Vector([1., -2., 1])
        self.assertEqual(np.array_str(x.components), str(x))

    def test_add(self):
        """
        Testing if override + operator works!
        :return:
        """

        x = Vector([1, 2, 3.])
        y = Vector([0., -2., 1.])
        self.assertEqual(Vector([1, 0., 4.]), x + y)

        ################################################################################################################

        x = Vector([-3., 2, 4.])
        y = Vector([0., -3., 1.])
        self.assertEqual(Vector([-3, -1., 5.]), x + y)

        ################################################################################################################

        x = Vector([-3.])
        y = Vector([-1, 2])

        with self.assertRaises(ArithmeticError):
            x + y

    def test_sub(self):
        """
        Testing if override - operator works!
        :return:
        """

        x = Vector([1, 2, 3.])
        y = Vector([0., -2., 1.])
        self.assertEqual(Vector([1, 4., 2.]), x - y)

        ################################################################################################################

        x = Vector([-3., 0., 4.])
        y = Vector([0., -3., 5.])
        self.assertEqual(Vector([-3, 3., -1.]), x - y)

        ################################################################################################################

        x = Vector([-3.])
        y = Vector([-1, 2])

        with self.assertRaises(ArithmeticError):
            x - y

    def test_magnitude(self):
        """
        Testing magnitude property
        :return:
        """

        x = Vector([1, 2, 3.])
        self.assertEqual(math.sqrt((1 * 1) + (2 * 2) + (3 * 3)), x.magnitude)

        ################################################################################################################

        x = Vector([1., -2.])
        self.assertEqual(math.sqrt((1 * 1) + (-2 * -2)), x.magnitude)

        ################################################################################################################

        x = Vector([rnd.uniform(-100., 100.) for _ in range(6)])
        self.assertEqual(math.sqrt(sum(component ** 2 for component in x.components)), x.magnitude)

    def test_similar(self):
        """
        Testing if two vectors are similar
        :return:
        """

        x = Vector([1, 2, 3, 4])
        y = deepcopy(x)
        self.assertTrue(Vector.all_close(x, y))

        ################################################################################################################

        x = Vector([1, .3])
        y = Vector([1, .3])
        self.assertTrue(Vector.all_close(x, y))

        ################################################################################################################

        x = Vector([1.2, 10.00001])
        y = Vector([1.20001, 10.])
        self.assertTrue(Vector.all_close(x, y))

        ################################################################################################################

        x = Vector([1, .3])
        y = Vector([1])
        self.assertFalse(Vector.all_close(x, y))

        ################################################################################################################

        x = Vector([1, .3])
        y = Vector([1, 4])
        self.assertFalse(Vector.all_close(x, y))

        ################################################################################################################

        x = Vector([1, .3])
        y = Vector([2, .3])
        self.assertFalse(Vector.all_close(x, y))

        ################################################################################################################

        x = Vector([1.2, 10.001])
        y = Vector([1.20001, 10.])
        self.assertFalse(Vector.all_close(x, y))

    def test_dominance(self):
        """
        Testing dominance function
        :return:
        """

        x = Vector([1, 2, 3])
        y = Vector([4, 5, 6])

        self.assertEqual(Dominance.is_dominated, Vector.dominance(x, y))
        self.assertEqual(Dominance.dominate, Vector.dominance(y, x))

        ################################################################################################################

        x = Vector([10, -1])
        y = Vector([2, -1])

        self.assertEqual(Dominance.dominate, Vector.dominance(x, y))
        self.assertEqual(Dominance.is_dominated, Vector.dominance(y, x))

        ################################################################################################################

        x = Vector([1, 2])
        y = Vector([0, 3])

        self.assertEqual(Dominance.otherwise, Vector.dominance(x, y))

        ################################################################################################################

        x = Vector([1.2, 10.00001])
        y = Vector([1.20001, 10.])

        # Are similar
        self.assertEqual(Dominance.equals, Vector.dominance(x, y))

        ################################################################################################################

        y = deepcopy(x)

        # Are equals
        self.assertEqual(Dominance.equals, Vector.dominance(x, y))

    def test_m3_max(self):
        """
        Testing m3_max function
        :return:
        """

        # Check first quadrant
        vectors = self.first_quadrant
        non_dominated = Vector.m3_max(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        dominance = TestVectors.check_if_all_are_non_dominated(vectors=vectors, non_dominated=non_dominated)

        self.assertTrue(dominance)

        ################################################################################################################

        # Check second quadrant
        vectors = self.second_quadrant
        non_dominated = Vector.m3_max(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        dominance = TestVectors.check_if_all_are_non_dominated(vectors=vectors, non_dominated=non_dominated)

        self.assertTrue(dominance)

        ################################################################################################################

        # Check third quadrant
        vectors = self.third_quadrant
        non_dominated = Vector.m3_max(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        dominance = TestVectors.check_if_all_are_non_dominated(vectors=vectors, non_dominated=non_dominated)

        self.assertTrue(dominance)

        ################################################################################################################

        # Check fourth quadrant
        vectors = self.fourth_quadrant
        non_dominated = Vector.m3_max(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        dominance = TestVectors.check_if_all_are_non_dominated(vectors=vectors, non_dominated=non_dominated)

        self.assertTrue(dominance)

        ################################################################################################################

        # Check all quadrants
        vectors = self.first_quadrant + self.second_quadrant + self.third_quadrant + self.fourth_quadrant
        non_dominated = Vector.m3_max(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        dominance = TestVectors.check_if_all_are_non_dominated(vectors=vectors, non_dominated=non_dominated)

        self.assertTrue(dominance)

    def test_m3_max_integer(self):
        """
        Testing m3_max function
        :return:
        """

        # Check first quadrant
        vectors = [(vector * (1 / Vector.relative)) for vector in self.first_quadrant]
        non_dominated = Vector.m3_max(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        dominance = TestVectors.check_if_all_are_non_dominated(vectors=vectors, non_dominated=non_dominated)

        self.assertTrue(dominance)

        ################################################################################################################

        # Check second quadrant
        vectors = self.second_quadrant
        non_dominated = Vector.m3_max(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        dominance = TestVectors.check_if_all_are_non_dominated(vectors=vectors, non_dominated=non_dominated)

        self.assertTrue(dominance)

        ################################################################################################################

        # Check third quadrant
        vectors = self.third_quadrant
        non_dominated = Vector.m3_max(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        dominance = TestVectors.check_if_all_are_non_dominated(vectors=vectors, non_dominated=non_dominated)

        self.assertTrue(dominance)

        ################################################################################################################

        # Check fourth quadrant
        vectors = self.fourth_quadrant
        non_dominated = Vector.m3_max(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        dominance = TestVectors.check_if_all_are_non_dominated(vectors=vectors, non_dominated=non_dominated)

        self.assertTrue(dominance)

        ################################################################################################################

        # Check all quadrants
        vectors = self.first_quadrant + self.second_quadrant + self.third_quadrant + self.fourth_quadrant
        non_dominated = Vector.m3_max(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        dominance = TestVectors.check_if_all_are_non_dominated(vectors=vectors, non_dominated=non_dominated)

        self.assertTrue(dominance)

    def test_m3_max_2_sets(self):
        """
        Testing m3_max function
        :return:
        """

        # Check first quadrant
        vectors = self.first_quadrant

        # Get m3_max_2_sets results
        non_dominated, dominated = Vector.m3_max_2_sets(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        check_non_dominated = TestVectors.check_if_all_are_non_dominated(vectors, non_dominated)
        self.assertTrue(check_non_dominated)

        # Check that each dominated vector is dominated at least by another vector
        check_dominated = TestVectors.check_if_each_vector_is_dominated_at_least_by_another(vectors, dominated)
        self.assertTrue(check_dominated)

        ################################################################################################################

        # Check second quadrant
        vectors = self.second_quadrant

        # Get m3_max_2_sets results
        non_dominated, dominated = Vector.m3_max_2_sets(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        check_non_dominated = TestVectors.check_if_all_are_non_dominated(vectors, non_dominated)
        self.assertTrue(check_non_dominated)

        # Check that each dominated vector is dominated at least by another vector
        check_dominated = TestVectors.check_if_each_vector_is_dominated_at_least_by_another(vectors, dominated)
        self.assertTrue(check_dominated)

        ################################################################################################################

        # Check third quadrant
        vectors = self.third_quadrant

        # Get m3_max_2_sets results
        non_dominated, dominated = Vector.m3_max_2_sets(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        check_non_dominated = TestVectors.check_if_all_are_non_dominated(vectors, non_dominated)
        self.assertTrue(check_non_dominated)

        # Check that each dominated vector is dominated at least by another vector
        check_dominated = TestVectors.check_if_each_vector_is_dominated_at_least_by_another(vectors, dominated)
        self.assertTrue(check_dominated)

        ################################################################################################################

        # Check fourth quadrant
        vectors = self.fourth_quadrant

        # Get m3_max_2_sets results
        non_dominated, dominated = Vector.m3_max_2_sets(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        check_non_dominated = TestVectors.check_if_all_are_non_dominated(vectors, non_dominated)
        self.assertTrue(check_non_dominated)

        # Check that each dominated vector is dominated at least by another vector
        check_dominated = TestVectors.check_if_each_vector_is_dominated_at_least_by_another(vectors, dominated)
        self.assertTrue(check_dominated)

        ################################################################################################################

        # Check all quadrants
        vectors = self.first_quadrant + self.second_quadrant + self.third_quadrant + self.fourth_quadrant

        # Get m3_max_2_sets results
        non_dominated, dominated = Vector.m3_max_2_sets(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        check_non_dominated = TestVectors.check_if_all_are_non_dominated(vectors, non_dominated)
        self.assertTrue(check_non_dominated)

        # Check that each dominated vector is dominated at least by another vector
        check_dominated = TestVectors.check_if_each_vector_is_dominated_at_least_by_another(vectors, dominated)
        self.assertTrue(check_dominated)

    def test_m3_max_2_sets_with_repetitions(self):
        """
        Testing m3_max function
        :return:
        """

        # Check first quadrant
        vectors = self.first_quadrant

        # Get m3_max_2_sets results
        non_dominated_unique, dominated, non_dominated_repeated = Vector.m3_max_2_sets_with_repetitions(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        check_non_dominated = TestVectors.check_if_all_are_non_dominated(vectors,
                                                                         non_dominated_unique + non_dominated_repeated)
        self.assertTrue(check_non_dominated)

        # Check that each dominated vector is dominated at least by another vector
        check_dominated = TestVectors.check_if_each_vector_is_dominated_at_least_by_another(vectors, dominated)
        self.assertTrue(check_dominated)

        ################################################################################################################

        # Check second quadrant
        vectors = self.second_quadrant

        # Get m3_max_2_sets results
        non_dominated, dominated = Vector.m3_max_2_sets(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        check_non_dominated = TestVectors.check_if_all_are_non_dominated(vectors,
                                                                         non_dominated_unique + non_dominated_repeated)
        self.assertTrue(check_non_dominated)

        # Check that each dominated vector is dominated at least by another vector
        check_dominated = TestVectors.check_if_each_vector_is_dominated_at_least_by_another(vectors, dominated)
        self.assertTrue(check_dominated)

        ################################################################################################################

        # Check third quadrant
        vectors = self.third_quadrant

        # Get m3_max_2_sets results
        non_dominated, dominated = Vector.m3_max_2_sets(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        check_non_dominated = TestVectors.check_if_all_are_non_dominated(vectors,
                                                                         non_dominated_unique + non_dominated_repeated)
        self.assertTrue(check_non_dominated)

        # Check that each dominated vector is dominated at least by another vector
        check_dominated = TestVectors.check_if_each_vector_is_dominated_at_least_by_another(vectors, dominated)
        self.assertTrue(check_dominated)

        ################################################################################################################

        # Check fourth quadrant
        vectors = self.fourth_quadrant

        # Get m3_max_2_sets results
        non_dominated, dominated = Vector.m3_max_2_sets(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        check_non_dominated = TestVectors.check_if_all_are_non_dominated(vectors,
                                                                         non_dominated_unique + non_dominated_repeated)
        self.assertTrue(check_non_dominated)

        # Check that each dominated vector is dominated at least by another vector
        check_dominated = TestVectors.check_if_each_vector_is_dominated_at_least_by_another(vectors, dominated)
        self.assertTrue(check_dominated)

        ################################################################################################################

        # Check all quadrants
        vectors = self.first_quadrant + self.second_quadrant + self.third_quadrant + self.fourth_quadrant

        # Get m3_max_2_sets results
        non_dominated, dominated = Vector.m3_max_2_sets(vectors=vectors)

        # Check that no vector of non_dominated list is dominated by other vector of total vectors.
        check_non_dominated = TestVectors.check_if_all_are_non_dominated(vectors,
                                                                         non_dominated_unique + non_dominated_repeated)
        self.assertTrue(check_non_dominated)

        # Check that each dominated vector is dominated at least by another vector
        check_dominated = TestVectors.check_if_each_vector_is_dominated_at_least_by_another(vectors, dominated)
        self.assertTrue(check_dominated)

    @staticmethod
    def check_if_all_are_non_dominated(vectors, non_dominated):
        """
        Check that no vector of non_dominated list is dominated by other vector of total vectors.
        :param vectors:
        :param non_dominated:
        :return:
        """
        return all([
            all([
                Vector.dominance(v1=vector_i, v2=vector_j) != Dominance.is_dominated for vector_j in vectors
            ]) for vector_i in non_dominated
        ])

    @staticmethod
    def check_if_each_vector_is_dominated_at_least_by_another(vectors, dominated):
        """
        Check that each dominated vector is dominated at least by another vector
        :param vectors:
        :param dominated:
        :return:
        """
        return all([
            any([
                Vector.dominance(v1=vector_i, v2=vector_j) == Dominance.is_dominated for vector_j in vectors
            ]) for vector_i in dominated
        ])
