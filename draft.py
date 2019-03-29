# import utils.pareto as up
import numpy as np

from models import Vector, VectorFloat

x = Vector([1, 2, 3])
y = Vector([4, 5, 6])
w = VectorFloat([1, 2, 3])
t = VectorFloat([4, 5, 6])

w + y

print(x ** y)

print(np.subtract(x, 3))

print(np.add(x, 2))
pass
