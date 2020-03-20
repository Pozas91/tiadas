"""
Example of backtracking agent, one without precision (full decimals) and one with decimal precision.
"""
from agents import AgentB
from environments import DeepSeaTreasureRightDownStochastic
from models import Vector


def draft_b(columns: int):
    # Show information
    print('Executing agent_b...')

    # Create environment
    environment = DeepSeaTreasureRightDownStochastic(columns=columns)

    # Create instance of AgentB
    agent = AgentB(environment=environment, limited_precision=False)

    # Does a simulations
    agent.simulate()

    return agent


def draft_b_lp(columns: int):
    # Show information
    print('Executing agent_b_lp...')

    # Create environment
    environment = DeepSeaTreasureRightDownStochastic(columns=columns)

    # Vector precision
    Vector.set_decimal_precision(decimal_precision=0.000001)

    # Create instance of AgentB
    agent = AgentB(environment=environment, limited_precision=True)

    # Does a simulations
    agent.simulate()

    return agent


def user_input():
    while True:
        try:
            value = int(input('How many columns[1, 10] do you want use? '))
        except ValueError:
            print('Not an integer! Please indicate a integer between 1 and 10.')
            continue
        else:
            return value


def main():
    # Retrieve user input
    columns = user_input()

    agent_b = draft_b(columns=columns)
    agent_b_lp = draft_b_lp(columns=columns)

    print('Agent B: ')
    print(agent_b.states_vectors)

    print('--------------------')

    print('Agent B LP: ')
    print(agent_b_lp.states_vectors)


if __name__ == '__main__':
    main()
