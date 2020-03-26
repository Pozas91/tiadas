"""
In this file we define methods to help us with the movements in the environments.
"""


def is_on_right_or_same_position(position: tuple, next_position: tuple):
    """
    Check if the position is on right or the same position that next position
    :param position:
    :param next_position:
    :return:
    """
    return is_on_right(position=position, next_position=next_position) or position == next_position


def is_on_right(position: tuple, next_position: tuple):
    """
    Check if the position is on right that next position
    :param position:
    :param next_position:
    :return:
    """

    # Decompose position
    x, y = position

    return (x + 1, y) == next_position


def is_on_down_or_same_position(position: tuple, next_position: tuple):
    """
    Check if the position is on down or the same position that next position
    :param position:
    :param next_position:
    :return:
    """
    return is_on_down(position=position, next_position=next_position) or position == next_position


def is_on_down(position: tuple, next_position: tuple):
    """
    Check if the position is on down that next position
    :param position:
    :param next_position:
    :return:
    """
    # Decompose position
    x, y = position

    return (x, y + 1) == next_position


def is_on_left_or_same_position(position: tuple, next_position: tuple):
    """
    Check if the position is on left or the same position that next position
    :param position:
    :param next_position:
    :return:
    """
    return is_on_left(position=position, next_position=next_position) or position == next_position


def is_on_left(position: tuple, next_position: tuple):
    """
    Check if the position is on left that next position
    :param position:
    :param next_position:
    :return:
    """
    # Decompose position
    x, y = position

    return (x - 1, y) == next_position


def is_on_up_or_same_position(position: tuple, next_position: tuple):
    """
    Check if the position is on up or the same position that next position
    :param position:
    :param next_position:
    :return:
    """
    return is_on_up(position=position, next_position=next_position) or position == next_position


def is_on_up(position: tuple, next_position: tuple):
    """
    Check if the position is on up that next position
    :param position:
    :param next_position:
    :return:
    """
    # Decompose position
    x, y = position

    return (x, y - 1) == next_position


def move_up(y: int, limit: int) -> int:
    """
    Move to up
    :param y:
    :param limit:
    :return:
    """
    return (y if y > 0 else limit) - 1


def move_right(x: int, limit: int) -> int:
    """
    Move to right
    :param x:
    :param limit:
    :return:
    """
    return (x + 1) % limit


def move_down(y: int, limit: int) -> int:
    """
    Move to down
    :param y:
    :param limit:
    :return:
    """
    return (y + 1) % limit


def move_left(x: int, limit: int) -> int:
    """
    Move to left
    :param x:
    :param limit:
    :return:
    """
    return (x if x > 0 else limit) - 1
