"""
Enum class for do easiest work with different evaluation mechanisms.
"""
from enum import Enum


class EvaluationMechanism(Enum):
    C = 'C'
    PO = 'PO'
    HV = 'HV'
    CHV = 'CHV'
    SCALARIZED = 'Linear Scalarized'

    def __str__(self):
        """
        Convert the agent type to a string
        :return:
        """
        return str(self.value)

    def __repr__(self):
        """
        Return a string that represent the agent type
        :return:
        """
        return str(self.value)

    @staticmethod
    def from_string(evaluation_mechanism: str):
        """
        Return a type of agent from the string given
        :param evaluation_mechanism:
        :return:
        """
        if evaluation_mechanism == str(EvaluationMechanism.C.value):
            result = EvaluationMechanism.C
        elif evaluation_mechanism == str(EvaluationMechanism.PO.value):
            result = EvaluationMechanism.PO
        elif evaluation_mechanism == str(EvaluationMechanism.HV.value):
            result = EvaluationMechanism.HV
        elif evaluation_mechanism == str(EvaluationMechanism.SCALARIZED.value):
            result = EvaluationMechanism.SCALARIZED
        else:
            raise ValueError('Unknown evaluation mechanism')

        return result
