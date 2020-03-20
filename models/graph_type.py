"""
Enum class for do easiest work with different graph types.
"""
from enum import Enum


class GraphType(Enum):
    STEPS = 'steps'
    EPISODES = 'episodes'
    TIME = 'time'
    MEMORY = 'memory'
    DATA_PER_STATE = 'data_per_state'
    SWEEP = 'sweep'
    V_S_0 = 'v_s_0'

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
    def from_string(graph_type: str):
        """
        Return a type of agent from the string given
        :param graph_type:
        :return:
        """

        if graph_type == str(GraphType.STEPS.value):
            result = GraphType.STEPS
        elif graph_type == str(GraphType.EPISODES.value):
            result = GraphType.EPISODES
        elif graph_type == str(GraphType.MEMORY.value):
            result = GraphType.MEMORY
        elif graph_type == str(GraphType.DATA_PER_STATE.value):
            result = GraphType.DATA_PER_STATE
        elif graph_type == str(GraphType.TIME.value):
            result = GraphType.TIME
        elif graph_type == str(GraphType.SWEEP.value):
            result = GraphType.SWEEP
        elif graph_type == str(GraphType.V_S_0.value):
            result = GraphType.V_S_0
        else:
            raise ValueError('Unknown graph type')

        return result
