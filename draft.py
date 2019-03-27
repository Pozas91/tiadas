# import utils.pareto as up
import numpy as np

from models import Vector
import operator

x = Vector([1, 2, 3])
y = Vector([4, 5, 6])
z = Vector([7, 8])
w = Vector([1, 2.45, 0.512])

print(x ** y)

print(np.subtract(x, 3))

print(np.add(x, 2))

print(w.to_int())

relative = Vector.relative

fourth_quadrant = (
    [
        # Problem
        Vector([2, -1]),
        Vector([3, -2]),
        Vector([1, -4]),
        Vector([3, -5]),
        Vector([5, -6]),
        Vector([7, -3]),
        Vector([10, -1]),

        # Repeats
        Vector([2, -1]),
        Vector([10, -1]),

        # Similar
        Vector([7 + relative, -3 - relative]),
        Vector([10 + relative, -1 + relative]),
    ],
    [
        # Non-dominated
        Vector([10, -1])
    ],
    [
        # Dominated
        Vector([2, -1]),
        Vector([3, -2]),
        Vector([1, -4]),
        Vector([3, -5]),
        Vector([5, -6]),
        Vector([7, -3]),
    ]
)
pass
