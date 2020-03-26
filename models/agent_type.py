"""
Enum class for do easiest work with different agents.
"""
from enum import Enum


class AgentType(Enum):
    B = 'b'
    PQL = 'pql'
    SCALARIZED = 'scalarized'
    PQL_EXP = 'pql_exp'
    PQL_EXP_3 = 'pql_exp_3'
    MPQ = 'mpq'
    W = 'w'

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
    def from_string(agent_type: str):
        """
        Return a type of agent from the string given
        :param agent_type:
        :return:
        """
        if agent_type == str(AgentType.B.value):
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
