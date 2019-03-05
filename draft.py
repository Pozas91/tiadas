import numpy as np

points = [(-1, 1), (-8, 8), (-9, 16), (-7, 5), (-17, 74), (-13, 24), (-19, 124), (-3, 2), (-14, 50), (-5, 3)]


def optimize():
    return points.pop()


def algorithm(p: (float, float), q: (float, float)) -> list:
    """
    Return a list of supported solutions costs
    :param p: 2D point
    :param q: 2D point
    :return:
    """

    # A new list with p and q
    result = [p, q]

    # Create a new stack
    accumulate = list()

    # Push a vector with p and q in the stack
    accumulate.append((p, q))

    while len(accumulate) > 0:
        # Pop the next pair of points from the stack.
        a, b = accumulate.pop()

        # Decompose points
        a_x, a_y = a
        b_x, b_y = b

        # Calculate the parameters of the new linear objective function (multiply by -1. to convert in maximize problem)
        w1 = np.multiply(a_y - b_y, -1.)
        w2 = np.multiply(b_x - a_x, -1.)

        # Solve P to find a new solution ang get its cost vector c.
        c = optimize()

        # Decompose c vector.
        c_x, c_y = c

        if (w1 * a_x + w2 * a_y) != (w1 * c_x + w2 * c_y):
            # c is the cost of a new supported solution

            # Push new pair in the stack
            accumulate.append((a, c))

            # Push new pair in the stack
            accumulate.append((c, b))

            # Add c to the result
            result.append(c)

    return result


def main():
    algorithm(p=(-1, 1), q=(-19, 120))


if __name__ == '__main__':
    main()
