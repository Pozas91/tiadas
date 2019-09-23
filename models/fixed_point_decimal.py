import decimal

from models.vector import Vector


class FixedPointDecimal(decimal.Decimal):

    def _fix(self, context):
        return super()._fix(context)._rescale(-Vector.decimals_allowed, context.rounding)


decimal.Decimal = FixedPointDecimal
