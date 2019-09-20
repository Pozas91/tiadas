from decorators import Singleton


@Singleton
class VectorConfiguration:
    # Number of decimals allowed by int numbers
    decimals_allowed = 2

    # Relative margin to compare of similarity of two elements
    relative_tolerance = 0.01
    absolute_tolerance = 0

    @staticmethod
    def set_absolute_tolerance(absolute_tolerance: float = 0.0, integer_mode: bool = False):
        multiply_factor = (10 ** VectorConfiguration.instance().decimals_allowed) if integer_mode else 1
        VectorConfiguration.instance().absolute_tolerance = absolute_tolerance * multiply_factor
