from enum import Enum


class AgentType(Enum):
    A1 = 'a1'
    B = 'b'
    PQL = 'pql'
    SCALARIZED = 'scalarized'
    PQL_EXP = 'pql_exp'
    PQL_EXP_3 = 'pql_exp_3'
    MPQ = 'mpq'
    W = 'w'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    @staticmethod
    def from_string(agent_type: str):

        if agent_type == str(AgentType.A1.value):
            result = AgentType.A1
        elif agent_type == str(AgentType.B.value):
            result = AgentType.B
        elif agent_type == str(AgentType.PQL.value):
            result = AgentType.PQL
        elif agent_type == str(AgentType.SCALARIZED.value):
            result = AgentType.SCALARIZED
        elif agent_type == str(AgentType.W.value):
            result = AgentType.W
        elif agent_type == str(AgentType.MPQ.value):
            result = AgentType.MPQ
        else:
            raise ValueError('Unknown agent type')

        return result
