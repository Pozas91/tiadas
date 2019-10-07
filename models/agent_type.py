from enum import Enum


class AgentType(Enum):
    A1 = 'a1'
    PQL = 'pql'
    SCALARIZED = 'scalarized'
    PQL_EXP = 'pql_exp'
    PQL_EXP_3 = 'pql_exp_3'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    @staticmethod
    def from_string(agent_type: str):

        if agent_type == str(AgentType.A1.value):
            result = AgentType.A1
        elif agent_type == str(AgentType.PQL.value):
            result = AgentType.PQL
        elif agent_type == str(AgentType.SCALARIZED.value):
            result = AgentType.SCALARIZED
        else:
            raise ValueError('Unknown agent type')

        return result
