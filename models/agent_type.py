from enum import Enum


class AgentType(Enum):
    A1 = 'a1'
    PQL = 'pql'
    SCALARIZED = 'scalarized'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    @staticmethod
    def from_string(graph_type: str):

        if graph_type == str(AgentType.A1.value):
            result = AgentType.A1
        elif graph_type == str(AgentType.PQL.value):
            result = AgentType.PQL
        else:
            result = AgentType.SCALARIZED

        return result
