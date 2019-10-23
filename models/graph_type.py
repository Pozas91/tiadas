from enum import Enum


class GraphType(Enum):
    STEPS = 'steps'
    EPISODES = 'episodes'
    TIME = 'time'
    MEMORY = 'memory'
    DATA_PER_STATE = 'data_per_state'
    SWEEP = 'sweep'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    @staticmethod
    def from_string(graph_type: str):

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
        else:
            raise ValueError('Unknown graph type')

        return result
