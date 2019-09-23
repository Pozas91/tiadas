from enum import Enum


class GraphType(Enum):
    STEPS = 'steps'
    EPISODES = 'episodes'
    TIME = 'time'
    MEMORY = 'memory'
    VECTORS_PER_CELL = 'vectors_per_cell'

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
        elif graph_type == str(GraphType.VECTORS_PER_CELL.value):
            result = GraphType.VECTORS_PER_CELL
        else:
            result = GraphType.TIME

        return result
