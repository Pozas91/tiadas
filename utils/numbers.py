from decimal import Decimal as D

from models import Vector


def are_equal_two_decimal_numbers(a: float, b: float) -> bool:
    """
    Check if two float numbers are equals using Decimal class
    :param a:
    :param b:
    :return:
    """
    return D(a).quantize(Vector.decimal_exponent) == D(b).quantize(Vector.decimal_exponent)
