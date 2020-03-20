from agents import AgentB
from environments import DeepSeaTreasureRightDownStochastic
from models import Vector


def draft_b(columns: int):
    # Create environment
    environment = DeepSeaTreasureRightDownStochastic(columns=columns)

    # Create instance of AgentB
    agent = AgentB(environment=environment, limited_precision=False)

    agent.simulate()


def draft_b_lp(columns: int):
    # Create environment
    environment = DeepSeaTreasureRightDownStochastic(columns=columns)

    # Vector precision
    Vector.set_decimal_precision(decimal_precision=0.000001)

    # Create instance of AgentB
    agent = AgentB(environment=environment, limited_precision=True)

    agent.simulate()


def main():
    columns = 2

    draft_b(columns=columns)
    draft_b_lp(columns=columns)


if __name__ == '__main__':
    main()
