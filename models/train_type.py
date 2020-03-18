from enum import Enum


class TrainType(Enum):
    TIME = 'time'

    def __str__(self):
        return str(self.value)

    def __repr__(self):
        return str(self.value)

    @staticmethod
    def from_string(agent_type: str):

        if agent_type == str(TrainType.TIME):
            result = TrainType.TIME
        else:
            raise ValueError('Unknown train type')

        return result
