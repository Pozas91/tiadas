from enum import Enum


class GraphType(Enum):
    STEPS = 'steps'
    EPOCHS = 'epochs'
    TIME = 'time'
    MEMORY = 'memory'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    @staticmethod
    def from_string(graph_type: str):

        if graph_type == str(GraphType.STEPS.value):
            result = GraphType.STEPS
        elif graph_type == str(GraphType.EPOCHS.value):
            result = GraphType.EPOCHS
        elif graph_type == str(GraphType.MEMORY.value):
            result = GraphType.MEMORY
        else:
            result = GraphType.TIME

        return result
