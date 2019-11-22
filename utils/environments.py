def is_on_right_or_same_position(state: tuple, next_state: tuple):
    return is_on_right(state=state, next_state=next_state) or state == next_state


def is_on_right(state: tuple, next_state: tuple):
    # Decompose state
    x, y = state

    return (x + 1, y) == next_state


def is_on_down_or_same_position(state: tuple, next_state: tuple):
    return is_on_down(state=state, next_state=next_state) or state == next_state


def is_on_down(state: tuple, next_state: tuple):
    # Decompose state
    x, y = state

    return (x, y + 1) == next_state


def is_on_left_or_same_position(state: tuple, next_state: tuple):
    return is_on_left(state=state, next_state=next_state) or state == next_state


def is_on_left(state: tuple, next_state: tuple):
    # Decompose state
    x, y = state

    return (x - 1, y) == next_state


def is_on_up_or_same_position(state: tuple, next_state: tuple):
    return is_on_up(state=state, next_state=next_state) or state == next_state


def is_on_up(state: tuple, next_state: tuple):
    # Decompose state
    x, y = state

    return (x, y - 1) == next_state


def move_up(y: int, limit: int = 5) -> int:
    """
    Move to up
    :param y:
    :param limit:
    :return:
    """
    return (y if y > 0 else limit) - 1


def move_right(x: int, limit: int = 13) -> int:
    """
    Move to right
    :param x:
    :param limit:
    :return:
    """
    return (x + 1) % limit


def move_down(y: int, limit: int = 5) -> int:
    """
    Move to down
    :param y:
    :param limit:
    :return:
    """
    return (y + 1) % limit


def move_left(x: int, limit: int = 13) -> int:
    """
    Move to left
    :param x:
    :param limit:
    :return:
    """
    return (x if x > 0 else limit) - 1
