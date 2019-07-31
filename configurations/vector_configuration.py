from decorators import Singleton


@Singleton
class VectorConfiguration:
    # Number of decimals allowed by int numbers
    decimals_allowed = 2

    # Relative margin to compare of similarity of two elements
    relative_tolerance = 0.01
    absolute_tolerance = 0
