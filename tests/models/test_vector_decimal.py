"""
Unit tests path where testing Vector model.
"""

import math
import random as rnd
import unittest
from copy import deepcopy

import numpy as np

from models import Dominance, VectorDecimal


class TestVectorFloat(unittest.TestCase):
    first_quadrant = None
    second_quadrant = None
    third_quadrant = None
    fourth_quadrant = None

    difference = 0.01

    def setUp(self):

        # Vector configuration
        VectorDecimal.set_decimal_precision(decimal_precision=2)

        self.first_quadrant = (
            [
                # Problem
                VectorDecimal([0, 6]),
                VectorDecimal([1, 6]),
                VectorDecimal([2, 5]),
                VectorDecimal([2, 4]),
                VectorDecimal([2, 2]),
                VectorDecimal([3, 4]),
                VectorDecimal([4, 3]),
                VectorDecimal([4, 1]),
                VectorDecimal([5, 3]),
                VectorDecimal([5, 2]),
                VectorDecimal([6, 0]),

                # Repeats
                VectorDecimal([0, 6]),
                VectorDecimal([4, 1]),

                # Similar
                VectorDecimal([5 + self.difference, 3 - self.difference]),
                VectorDecimal([2 + self.difference, 4 - self.difference]),
            ],
            [
                # Non-dominated VectorFloats
                VectorDecimal([1, 6]),
                VectorDecimal([2, 5]),
                VectorDecimal([3, 4]),
                VectorDecimal([5, 3]),
                VectorDecimal([6, 0])
            ],
            [
                # Dominated VectorFloats
                VectorDecimal([0, 6]),
                VectorDecimal([2, 4]),
                VectorDecimal([2, 2]),
                VectorDecimal([4, 3]),
                VectorDecimal([4, 1]),
                VectorDecimal([5, 2]),
            ]
        )

        self.second_quadrant = (
            [
                # Problem
                VectorDecimal([-1, 0]),
                VectorDecimal([-3, 4]),
                VectorDecimal([-4, 2]),
                VectorDecimal([-4, 7]),
                VectorDecimal([-6, 6]),
                VectorDecimal([-6, 0]),
                VectorDecimal([-8, 2]),

                # Repeats
                VectorDecimal([-1, 0]),
                VectorDecimal([-6, 6]),

                # Similar
                VectorDecimal([-4 + self.difference, 2 + self.difference]),
                VectorDecimal([-4 - self.difference, 7 + self.difference]),
            ],
            [
                # Non-dominated
                VectorDecimal([-1, 0]),
                VectorDecimal([-4, 7]),
                VectorDecimal([-3, 4]),
            ],
            [
                # Dominated VectorFloats
                VectorDecimal([-4, 2]),
                VectorDecimal([-6, 6]),
                VectorDecimal([-6, 0]),
                VectorDecimal([-8, 2]),
            ]
        )

        self.third_quadrant = (
            [
                # Problem
                VectorDecimal([-1, -4]),
                VectorDecimal([-2, -1]),
                VectorDecimal([-3, -6]),
                VectorDecimal([-4, -2]),
                VectorDecimal([-5, -4]),
                VectorDecimal([-7, -1]),

                # Repeats
                VectorDecimal([-1, -4]),
                VectorDecimal([-7, -1]),

                # Similar
                VectorDecimal([-2 - self.difference, -1 - self.difference]),
                VectorDecimal([-4 + self.difference, -2 + self.difference]),
            ],
            [
                # Non-dominated
                VectorDecimal([-2, -1]),
                VectorDecimal([-1, -4])
            ],
            [
                # Dominated VectorFloats
                VectorDecimal([-3, -6]),
                VectorDecimal([-4, -2]),
                VectorDecimal([-5, -4]),
                VectorDecimal([-7, -1]),
            ]
        )

        self.fourth_quadrant = (
            [
                # Problem
                VectorDecimal([2, -1]),
                VectorDecimal([3, -2]),
                VectorDecimal([1, -4]),
                VectorDecimal([3, -5]),
                VectorDecimal([5, -6]),
                VectorDecimal([7, -3]),
                VectorDecimal([10, -1]),

                # Repeats
                VectorDecimal([2, -1]),
                VectorDecimal([10, -1]),

                # Similar
                VectorDecimal([7 + self.difference, -3 - self.difference]),
                VectorDecimal([10 + self.difference, -1 + self.difference]),
            ],
            [
                # Non-dominated
                VectorDecimal([10, -1])
            ],
            [
                # Dominated
                VectorDecimal([2, -1]),
                VectorDecimal([3, -2]),
                VectorDecimal([1, -4]),
                VectorDecimal([3, -5]),
                VectorDecimal([5, -6]),
                VectorDecimal([7, -3]),
            ]
        )

        self.all_quadrants = (
            # Problem
            self.first_quadrant[0] + self.second_quadrant[0] + self.third_quadrant[0] + self.fourth_quadrant[0],
            [
                # Non-dominate
                VectorDecimal([-4, 7]),
                VectorDecimal([1, 6]),
                VectorDecimal([2, 5]),
                VectorDecimal([3, 4]),
                VectorDecimal([5, 3]),
                VectorDecimal([6, 0]),
                VectorDecimal([10, -1])
            ],
            [
                # Dominated
                VectorDecimal([0, 6]),
                VectorDecimal([2, 4]),
                VectorDecimal([2, 2]),
                VectorDecimal([4, 3]),
                VectorDecimal([4, 1]),
                VectorDecimal([5, 2]),
                VectorDecimal([-1, 0]),
                VectorDecimal([-3, 4]),
                VectorDecimal([-4, 2]),
                VectorDecimal([-6, 6]),
                VectorDecimal([-6, 0]),
                VectorDecimal([-8, 2]),
                VectorDecimal([-1, -4]),
                VectorDecimal([-2, -1]),
                VectorDecimal([-3, -6]),
                VectorDecimal([-4, -2]),
                VectorDecimal([-5, -4]),
                VectorDecimal([-7, -1]),
                VectorDecimal([2, -1]),
                VectorDecimal([1, -4]),
                VectorDecimal([3, -2]),
                VectorDecimal([3, -5]),
                VectorDecimal([5, -6]),
                VectorDecimal([7, -3]),
            ]
        )

    def tearDown(self):
        pass

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """

        components = [rnd.uniform(-100., 100.) for _ in range(rnd.randint(2, 10))]

        # List
        x = VectorDecimal(components)
        self.assertTrue(isinstance(x.components, np.ndarray))

        # ndarray
        x = VectorDecimal(np.asarray(components))
        self.assertTrue(isinstance(x.components, np.ndarray))

    def test_length(self):
        """
        Testing if override len() operator works!
        :return:
        """

        for _ in range(5):
            n = rnd.randint(1, 20)
            n_length = VectorDecimal([rnd.uniform(-100., 100.) for _ in range(n)])
            self.assertEqual(n, len(n_length))

    def test_equal(self):
        """
        Testing if override = operator works!
        :return:
        """

        x = VectorDecimal([rnd.uniform(-100., 100.) for _ in range(rnd.randint(2, 10))])
        y = deepcopy(x)

        self.assertEqual(y, x)

    def test_ge(self):
        """
        Testing if override >= operator works!
        :return:
        """

        x = VectorDecimal([5 + self.difference, 3 + self.difference])
        y = VectorDecimal([4, 4])
        z = VectorDecimal([5, 3])
        w = VectorDecimal([6, 4])
        t = VectorDecimal([3, 2])

        self.assertFalse(x >= y)
        self.assertTrue(x >= z)
        self.assertFalse(x >= w)
        self.assertTrue(x >= t)

        self.assertFalse(y >= x)
        self.assertFalse(y >= z)
        self.assertFalse(y >= w)
        self.assertTrue(y >= t)

        self.assertTrue(z >= x)
        self.assertFalse(z >= y)
        self.assertFalse(z >= w)
        self.assertTrue(z >= t)

        self.assertTrue(w >= x)
        self.assertTrue(w >= y)
        self.assertTrue(w >= z)
        self.assertTrue(w >= t)

        self.assertFalse(t >= x)
        self.assertFalse(t >= y)
        self.assertFalse(t >= z)
        self.assertFalse(t >= w)

    def test_gt(self):
        """
        Testing if override > operator works!
        :return:
        """

        x = VectorDecimal([5 + self.difference, 3 + self.difference])
        y = VectorDecimal([4, 4])
        z = VectorDecimal([5, 3])
        w = VectorDecimal([6, 4])
        t = VectorDecimal([3, 2])

        self.assertFalse(x > y)
        self.assertFalse(x > z)
        self.assertFalse(x > w)
        self.assertTrue(x > t)

        self.assertFalse(y > x)
        self.assertFalse(y > z)
        self.assertFalse(y > w)
        self.assertTrue(y > t)

        self.assertFalse(z > x)
        self.assertFalse(z > y)
        self.assertFalse(z > w)
        self.assertTrue(z > t)

        self.assertTrue(w > x)
        self.assertFalse(w > y)
        self.assertTrue(w > z)
        self.assertTrue(w > t)

        self.assertFalse(t > x)
        self.assertFalse(t > y)
        self.assertFalse(t > z)
        self.assertFalse(t > w)

    def test_lt(self):
        """
        Testing if override < operator works!
        :return:
        """

        x = VectorDecimal([5 + self.difference, 3 + self.difference])
        y = VectorDecimal([4, 4])
        z = VectorDecimal([5, 3])
        w = VectorDecimal([6, 4])
        t = VectorDecimal([3, 2])

        self.assertFalse(x < y)
        self.assertFalse(x < z)
        self.assertTrue(x < w)
        self.assertFalse(x < t)

        self.assertFalse(y < x)
        self.assertFalse(y < z)
        self.assertFalse(y < w)
        self.assertFalse(y < t)

        self.assertFalse(z < x)
        self.assertFalse(z < y)
        self.assertTrue(z < w)
        self.assertFalse(z < t)

        self.assertFalse(w < x)
        self.assertFalse(w < y)
        self.assertFalse(w < z)
        self.assertFalse(w < t)

        self.assertTrue(t < x)
        self.assertTrue(t < y)
        self.assertTrue(t < z)
        self.assertTrue(t < w)

    def test_le(self):
        """
        Testing if override < operator works!
        :return:
        """

        x = VectorDecimal([5 + self.difference, 3 + self.difference])
        y = VectorDecimal([4, 4])
        z = VectorDecimal([5, 3])
        w = VectorDecimal([6, 4])
        t = VectorDecimal([3, 2])

        self.assertFalse(x <= y)
        self.assertTrue(x <= z)
        self.assertTrue(x <= w)
        self.assertFalse(x <= t)

        self.assertFalse(y <= x)
        self.assertFalse(y <= z)
        self.assertTrue(y <= w)
        self.assertFalse(y <= t)

        self.assertTrue(z <= x)
        self.assertFalse(z <= y)
        self.assertTrue(z <= w)
        self.assertFalse(z <= t)

        self.assertFalse(w <= x)
        self.assertFalse(w <= y)
        self.assertFalse(w <= z)
        self.assertFalse(w <= t)

        self.assertTrue(t <= x)
        self.assertTrue(t <= y)
        self.assertTrue(t <= z)
        self.assertTrue(t <= w)

    def test_str(self):
        """
        Testing if override str operator works!
        :return:
        """

        x = VectorDecimal([1, 2, 3])
        self.assertEqual(np.array_str(x.components), str(x))

        ################################################################################################################

        x = VectorDecimal([1, -2])
        self.assertEqual(np.array_str(x.components), str(x))

        ################################################################################################################

        x = VectorDecimal([1., -2., 1])
        self.assertEqual(np.array_str(x.components), str(x))

    def test_add(self):
        """
        Testing if override + operator works!
        :return:
        """

        x = VectorDecimal([1, 2, 3.])
        y = VectorDecimal([0., -2., 1.])
        self.assertEqual(VectorDecimal([1, 0., 4.]), x + y)

        ################################################################################################################

        x = VectorDecimal([-3., 2, 4.])
        y = VectorDecimal([0., -3., 1.])
        self.assertEqual(VectorDecimal([-3, -1., 5.]), x + y)

        ################################################################################################################

        x = VectorDecimal([1, 2, 3])
        self.assertEqual(VectorDecimal([2, 3, 4]), x + 1)

        ################################################################################################################

        x = VectorDecimal([1, 2, 3])
        self.assertEqual(VectorDecimal([2, 3, 4]), x + 1.)

        ################################################################################################################

        x = VectorDecimal([1, 2, 3])
        y = VectorDecimal([4, 5, 6, 7])

        with self.assertRaises(ValueError):
            x + y
            y + x

    def test_sub(self):
        """
        Testing if override - operator works!
        :return:
        """

        x = VectorDecimal([1, 2, 3.])
        y = VectorDecimal([0., -2., 1.])
        self.assertEqual(VectorDecimal([1, 4., 2.]), x - y)

        ################################################################################################################

        x = VectorDecimal([-3., 0., 4.])
        y = VectorDecimal([0., -3., 5.])
        self.assertEqual(VectorDecimal([-3, 3., -1.]), x - y)

        ################################################################################################################

        x = VectorDecimal([1, 2, 3])
        self.assertEqual(VectorDecimal([0, 1, 2]), x - 1)

        ################################################################################################################

        x = VectorDecimal([1, 2, 3])
        self.assertEqual(VectorDecimal([0, 1, 2]), x - 1.)

        ################################################################################################################

        x = VectorDecimal([1, 2, 3])
        y = VectorDecimal([4, 5, 6, 7])

        with self.assertRaises(ValueError):
            x - y
            y - x

    def test_mul(self):
        """
        Testing if override * operator works!
        :return:
        """

        x = VectorDecimal([1, 2, 3.])
        y = VectorDecimal([0., -2., 1.])
        self.assertEqual(VectorDecimal([0, -4, 3]), x * y)

        ################################################################################################################

        x = VectorDecimal([-3., 2, 4.])
        y = VectorDecimal([0., -3., 1.])
        self.assertEqual(VectorDecimal([0, -6, 4]), x * y)

        ################################################################################################################

        x = VectorDecimal([-3., 2, 4.])
        self.assertEqual(VectorDecimal([-6, 4, 8]), x * 2)

        ################################################################################################################

        x = VectorDecimal([-3., 2, 4.])
        self.assertEqual(VectorDecimal([-6, 4, 8]), x * 2.)

        ################################################################################################################

        x = VectorDecimal([1, 2, 3])
        y = VectorDecimal([4, 5, 6, 7])

        with self.assertRaises(ValueError):
            x * y
            y * x

    def test_pow(self):
        """
        Testing if override ** operator works!
        :return:
        """

        x = VectorDecimal([1, 2, 3.])
        y = VectorDecimal([0., -2., 1.])
        self.assertEqual(VectorDecimal([1, 0.25, 3]), x ** y)

        ################################################################################################################

        x = VectorDecimal([-3., 2, 4.])
        y = VectorDecimal([0., -3., 1.])
        self.assertEqual(VectorDecimal([1, 0.125, 4]), x ** y)

        ################################################################################################################

        x = VectorDecimal([-3., 2, 4.])
        self.assertEqual(VectorDecimal([9, 4, 16]), x ** 2)

        ################################################################################################################

        x = VectorDecimal([-3., 2, 4.])
        self.assertEqual(VectorDecimal([9, 4, 16]), x ** 2.)

        ################################################################################################################

        x = VectorDecimal([1, 2, 3])
        y = VectorDecimal([4, 5, 6, 7])

        with self.assertRaises(ValueError):
            x ** y
            y ** x

    def test_magnitude(self):
        """
        Testing magnitude property
        :return:
        """

        x = VectorDecimal([1, 2, 3.])
        self.assertEqual(math.sqrt((1 * 1) + (2 * 2) + (3 * 3)), x.magnitude())

        ################################################################################################################

        x = VectorDecimal([1., -2.])
        self.assertEqual(math.sqrt((1 * 1) + (-2 * -2)), x.magnitude())

        ################################################################################################################

        x = VectorDecimal([rnd.uniform(-100., 100.) for _ in range(6)])
        self.assertEqual(math.sqrt(sum(component ** 2 for component in x.components)), x.magnitude())

    def test_all_close(self):
        """
        Testing if two Vectors are similar
        :return:
        """

        x = VectorDecimal([1, 2, 3, 4])
        y = deepcopy(x)
        self.assertTrue(VectorDecimal.all_close(x, y))

        ################################################################################################################

        x = VectorDecimal([1, .3])
        y = VectorDecimal([1, .3])
        self.assertTrue(VectorDecimal.all_close(x, y))

        ################################################################################################################

        x = VectorDecimal([1.2, 10 + self.difference])
        y = VectorDecimal([1.2 + self.difference, 10.])
        self.assertTrue(VectorDecimal.all_close(x, y))

        ################################################################################################################

        x = VectorDecimal([1.2 + self.difference, 10])
        y = VectorDecimal([1.2, 10. + self.difference])
        self.assertTrue(VectorDecimal.all_close(x, y))

        ################################################################################################################

        x = VectorDecimal([1, .3])
        y = VectorDecimal([1])
        self.assertTrue(VectorDecimal.all_close(x, y))

        ################################################################################################################

        x = VectorDecimal([1, .3])
        y = VectorDecimal([.3])
        self.assertFalse(VectorDecimal.all_close(x, y))

        ################################################################################################################

        x = VectorDecimal([1, .3])
        y = VectorDecimal([1, 4])
        self.assertFalse(VectorDecimal.all_close(x, y))

        ################################################################################################################

        x = VectorDecimal([1, .3])
        y = VectorDecimal([2, .3])
        self.assertFalse(VectorDecimal.all_close(x, y))

    def test_dominance(self):
        """
        Testing dominance function
        :return:
        """

        x = VectorDecimal([1, 2, 3])
        y = VectorDecimal([4, 5, 6])

        self.assertEqual(Dominance.is_dominated, VectorDecimal.dominance(x, y))
        self.assertEqual(Dominance.dominate, VectorDecimal.dominance(y, x))

        ################################################################################################################

        x = VectorDecimal([10, -1])
        y = VectorDecimal([2, -1])

        self.assertEqual(Dominance.dominate, VectorDecimal.dominance(x, y))
        self.assertEqual(Dominance.is_dominated, VectorDecimal.dominance(y, x))

        ################################################################################################################

        x = VectorDecimal([1, 2])
        y = VectorDecimal([0, 3])

        self.assertEqual(Dominance.otherwise, VectorDecimal.dominance(x, y))

        ################################################################################################################

        x = VectorDecimal([1.2, 10.00001])
        y = VectorDecimal([1.20001, 10.])

        # Are similar
        self.assertEqual(Dominance.equals, VectorDecimal.dominance(x, y))

        ################################################################################################################

        y = deepcopy(x)

        # Are equals
        self.assertEqual(Dominance.equals, VectorDecimal.dominance(x, y))

    def test_m3_max(self):
        """
        Testing m3_max function
        :return:
        """

        # Test problems
        for problem, solution, _ in [self.first_quadrant, self.second_quadrant, self.third_quadrant,
                                     self.fourth_quadrant, self.all_quadrants]:

            # Calc non_dominated Vectors
            non_dominated = VectorDecimal.m3_max(vectors=problem)

            # While not is empty
            while non_dominated:
                # Extract from non_dominated list and remove it from solution list
                solution.remove(non_dominated.pop())

            # After previous process if solution list have any element, then assert is failed.
            self.assertFalse(solution)

    def test_m3_max_2_lists(self):
        """
        Testing m3_max function
        :return:
        """

        # Prepare Vectors
        problems = [
            (
                # Problem
                self.first_quadrant[0],
                # Non-dominated uniques
                self.first_quadrant[1],
                # Dominated (duplicates included)
                self.first_quadrant[2] + [
                    VectorDecimal([2 + self.difference, 4 - self.difference]),
                    VectorDecimal([0, 6]), VectorDecimal([4, 1])
                ],
            ),
            (
                # Problem
                self.second_quadrant[0],
                # Non-dominated uniques
                self.second_quadrant[1],
                # Dominated (duplicates included)
                self.second_quadrant[2] + [VectorDecimal([-6, 6]),
                                           VectorDecimal([-4 + self.difference,
                                                          2 + self.difference])],
            ),
            (
                # Problem
                self.third_quadrant[0],
                # Non-dominated uniques
                self.third_quadrant[1],
                # Dominated (duplicates included)
                self.third_quadrant[2] + [VectorDecimal([-7, -1]),
                                          VectorDecimal([-4 + self.difference,
                                                         -2 + self.difference])],
            ),
            (
                # Problem
                self.fourth_quadrant[0],
                # Non-dominated uniques
                self.fourth_quadrant[1],
                # Dominated (duplicates included)
                self.fourth_quadrant[2] + [
                    VectorDecimal([7 + self.difference, -3 - self.difference]),
                    VectorDecimal([2, -1])],
            ),
            (
                # Problem
                self.all_quadrants[0],
                # Non-dominated uniques
                self.all_quadrants[1],
                # Dominated (duplicates included)
                self.all_quadrants[2] + [
                    VectorDecimal([7 + self.difference, -3 - self.difference]),
                    VectorDecimal([-7, -1]),
                    VectorDecimal([-4 + self.difference, -2 + self.difference]),
                    VectorDecimal([-6, 6]),
                    VectorDecimal([-4 + self.difference, 2 + self.difference]),
                    VectorDecimal([0, 6]),
                    VectorDecimal([4, 1]),
                    VectorDecimal([2 + self.difference, 4 - self.difference]),
                    VectorDecimal([2, -1]),
                    VectorDecimal([-2 - self.difference, -1 - self.difference]),
                    VectorDecimal([-1, -4]),
                    VectorDecimal([-1, 0])
                ],
            )
        ]

        # Test problems
        for problem, solution_non_dominated, solution_dominated in problems:

            # Apply m3_max_2_lists algorithm
            non_dominated, dominated = VectorDecimal.m3_max_2_lists(vectors=problem)

            # While not is empty
            while non_dominated:
                # Extract from non_dominated list and remove from solution list
                solution_non_dominated.remove(non_dominated.pop())

            # After previous process if solution list have any element, then assert is failed.
            self.assertFalse(solution_non_dominated)

            while dominated:
                # Extract from dominated list and remove from solution list
                solution_dominated.remove(dominated.pop())

            # After previous process if solution list have any element, then assert is failed.
            self.assertFalse(solution_dominated)

    def test_m3_max_2_lists_not_duplicates(self):
        """
        Testing m3_max function
        :return:
        """

        # Test problems
        for problem, solution_non_dominated, solution_dominated in [self.first_quadrant, self.second_quadrant,
                                                                    self.third_quadrant, self.fourth_quadrant,
                                                                    self.all_quadrants]:

            # Apply m3_max_2_lists algorithm
            non_dominated, dominated = VectorDecimal.m3_max_2_lists_not_duplicates(vectors=problem)

            # While not is empty
            while non_dominated:
                # Extract from non_dominated list and remove from solution list
                solution_non_dominated.remove(non_dominated.pop())

            # After previous process if solution list have any element, then assert is failed.
            self.assertFalse(solution_non_dominated)

            while dominated:
                # Extract from dominated list and remove from solution list
                solution_dominated.remove(dominated.pop())

            # After previous process if solution list have any element, then assert is failed.
            self.assertFalse(solution_dominated)

    def test_m3_max_2_lists_with_repetitions(self):
        """
        Testing m3_max function
        :return:
        """

        # Prepare Vectors
        problems = [
            (
                # Problem
                self.first_quadrant[0],
                # Non-dominated uniques
                self.first_quadrant[1],
                # Dominated (duplicates included)
                self.first_quadrant[2] + [
                    VectorDecimal([2 + self.difference, 4 - self.difference]),
                    VectorDecimal([0, 6]),
                    VectorDecimal([4, 1])
                ],
                # Non-dominated repeated
                [
                    VectorDecimal([5 + self.difference, 3 - self.difference])
                ]
            ),
            (
                # Problem
                self.second_quadrant[0],
                # Non-dominated uniques
                self.second_quadrant[1],
                # Dominated (duplicates included)
                self.second_quadrant[2] + [
                    VectorDecimal([-6, 6]),
                    VectorDecimal([-4 + self.difference, 2 + self.difference])
                ],
                # Non-dominated repeated
                [
                    VectorDecimal([-4 - self.difference, 7 + self.difference]),
                    VectorDecimal([-1, 0])
                ]
            ),
            (
                # Problem
                self.third_quadrant[0],
                # Non-dominated uniques
                self.third_quadrant[1],
                # Dominated (duplicates included)
                self.third_quadrant[2] + [
                    VectorDecimal([-7, -1]),
                    VectorDecimal([-4 + self.difference, -2 + self.difference])
                ],
                # Non-dominated repeated
                [
                    VectorDecimal([-2 - self.difference, -1 - self.difference]),
                    VectorDecimal([-1, -4])
                ]
            ),
            (
                # Problem
                self.fourth_quadrant[0],
                # Non-dominated uniques
                self.fourth_quadrant[1],
                # Dominated (duplicates included)
                self.fourth_quadrant[2] + [
                    VectorDecimal([7 + self.difference, -3 - self.difference]),
                    VectorDecimal([2, -1])
                ],
                # Non-dominated repeated
                [
                    VectorDecimal([10 + self.difference, -1 + self.difference]),
                    VectorDecimal([10, -1])
                ]
            ),
            (
                # Problem
                self.all_quadrants[0],
                # Non-dominated uniques
                self.all_quadrants[1],
                # Dominated (duplicates included)
                self.all_quadrants[2] + [
                    VectorDecimal([7 + self.difference, -3 - self.difference]),
                    VectorDecimal([-7, -1]),
                    VectorDecimal([-4 + self.difference, -2 + self.difference]),
                    VectorDecimal([-6, 6]),
                    VectorDecimal([-4 + self.difference, 2 + self.difference]),
                    VectorDecimal([0, 6]),
                    VectorDecimal([4, 1]), VectorDecimal([-1, 0]),
                    VectorDecimal([2 + self.difference, 4 - self.difference]),
                    VectorDecimal([2, -1]),
                    VectorDecimal([-2 - self.difference, -1 - self.difference]),
                    VectorDecimal([-1, -4]),
                ],
                # Non-dominated repeated
                [
                    VectorDecimal([10 + self.difference, -1 + self.difference]),
                    VectorDecimal([10, -1]),
                    VectorDecimal([-4 - self.difference, 7 + self.difference]),
                    VectorDecimal([5 + self.difference, 3 - self.difference])
                ]
            )
        ]

        for problem, solution_non_dominated_uniques, solution_dominated, solution_non_dominated_repeat in problems:
            # Apply m3_max_2_lists_with_repetitions algorithm
            non_dominated_unique, dominated, non_dominated_repeated = VectorDecimal.m3_max_2_lists_with_repetitions(
                vectors=problem)

            # While not is empty
            while non_dominated_unique:
                # Extract from non_dominated_unique list and remove from solution list
                solution_non_dominated_uniques.remove(non_dominated_unique.pop())

                # After previous process if solution list have any element, then assert is failed.
            self.assertFalse(solution_non_dominated_uniques)

            # While not is empty
            while dominated:
                # Extract from dominated list and remove from solution list
                solution_dominated.remove(dominated.pop())

            # After previous process if solution list have any element, then assert is failed.
            self.assertFalse(solution_dominated)

            # While not is empty
            while non_dominated_repeated:
                # Extract from non_dominated_repeat list and remove from solution list
                solution_non_dominated_repeat.remove(non_dominated_repeated.pop())

            # After previous process if solution list have any element, then assert is failed.
            self.assertFalse(solution_non_dominated_repeat)
