"""
Unit tests file where testing vector model.
"""

import random as rnd
import unittest
from copy import deepcopy

import numpy as np

from models import Vector


class TestVectors(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_init(self):
        """
        Testing if constructor works
        :return:
        """
        vector = Vector([rnd.uniform(-100., 100.) for _ in range(rnd.randint(2, 10))])
        self.assertTrue(isinstance(vector.components, np.ndarray))

    def test_length(self):
        """
        Testing if override len() operator works!
        :return:
        """
        two_length = Vector([rnd.uniform(-100., 100.) for _ in range(2)])
        self.assertEqual(len(two_length), 2)

        three_length = Vector([rnd.uniform(-100., 100.) for _ in range(3)])
        self.assertEqual(len(three_length), 3)

        six_length = Vector([rnd.uniform(-100., 100.) for _ in range(6)])
        self.assertEqual(len(six_length), 6)

    def test_equal(self):
        """
        Testing if override = operator works!
        :return:
        """

        x = Vector([rnd.uniform(-100., 100.) for _ in range(rnd.randint(2, 10))])
        y = deepcopy(x)

        self.assertEqual(x, y)

    def test_str(self):
        """
        Testing if override str operator works!
        :return:
        """

        x = Vector([1, 2, 3])
        self.assertEqual(str(x), '[1 2 3]')

        x = Vector([1, -2])
        self.assertEqual(str(x), '[1 -2]')

        x = Vector([1., -2., 1])
        self.assertEqual(str(x), '[1.0 -2.0 1.0]')

    def test_add(self):
        """
        Testing if override + operator works!
        :return:
        """

        x = Vector([1, 2, 3.])
        y = Vector([0., -2., 1.])

        self.assertEqual(x + y, Vector([1, 0., 4.]))

        x = Vector([-3., 2, 4.])
        y = Vector([0., -3., 1.])

        self.assertEqual(x + y, Vector([-3, -1., 5.]))

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

        self.assertEqual(x - y, Vector([1, 4., 2.]))

        x = Vector([-3., 0., 4.])
        y = Vector([0., -3., 5.])

        self.assertEqual(x - y, Vector([-3, 3., -1.]))

        x = Vector([-3.])
        y = Vector([-1, 2])

        with self.assertRaises(ArithmeticError):
            x - y
