from decimal import Decimal

from models import Vector


def are_equal_two_decimal_numbers(a: float, b: float) -> bool:
    """
    Check if two float numbers are equals using Decimal class
    :param a:
    :param b:
    :return:
    """
    return Decimal(a).quantize(Vector.decimal_precision) == Decimal(b).quantize(Vector.decimal_precision)


def round_with_precision(number: float, precision: Decimal) -> float:
    """
    Round a number with precision given.
    :param number:
    :param precision:
    :return:
    """
    n_of_decimals = abs(precision.as_tuple().exponent)
    precision = float(precision)
    return round(round(number / precision) * precision, n_of_decimals)
