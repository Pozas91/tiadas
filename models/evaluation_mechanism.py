from enum import Enum


class EvaluationMechanism(Enum):
    C = 'C'
    PO = 'PO'
    HV = 'HV'
    PARETO = 'Pareto'
    SCALARIZED = 'Linear Scalarized'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    @staticmethod
    def from_string(evaluation_mechanism: str):

        if evaluation_mechanism == str(EvaluationMechanism.C.value):
            result = EvaluationMechanism.C
        elif evaluation_mechanism == str(EvaluationMechanism.PO.value):
            result = EvaluationMechanism.PO
        elif evaluation_mechanism == str(EvaluationMechanism.HV.value):
            result = EvaluationMechanism.HV
        elif evaluation_mechanism == str(EvaluationMechanism.PARETO.value):
            result = EvaluationMechanism.PARETO
        else:
            result = EvaluationMechanism.SCALARIZED

        return result
