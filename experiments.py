from colorama import init

import utils.graphs as ug
from environments import BonusWorld
from models import GraphType, Vector, EvaluationMechanism, AgentType

init(autoreset=True)


def main():
    # Basic configuration
    alpha = 1
    number_of_agents = 1
    gamma = 1.
    max_steps = 250
    initial_state = (0, 0)
    evaluation_mechanism = EvaluationMechanism.CHV
    decimals_allowed = 7
    epsilon = 0.7
    states_to_observe = [((0, 0), False)]
    integer_mode = False
    solution = None

    # Environment configuration
    # environment = PressurizedBountifulSeaTreasure(initial_state=initial_state)
    # hv_reference = Vector([-25, 0, -120])
    environment = BonusWorld(initial_state=initial_state)
    # hv_reference = Vector([-25, 0])
    hv_reference = Vector([0, 0, -150])

    # Variable parameters
    variable = 'epsilon'
    # variable = 'evaluation_mechanism'

    agents_configuration = {
        # AgentType.A1: {
        # EvaluationMechanism.HV: 'yellow',
        # EvaluationMechanism.C:  'orange',
        # EvaluationMechanism.PO: 'blue',
        # 0.01: 'blue',
        # 0.03: 'cyan',
        # 0.1: 'black',
        # 0.3: 'gold',
        # 0.6: 'orange',
        # 0.8: 'fuchsia',
        # 1.0: 'cyan'
        # },
        AgentType.PQL: {
            # EvaluationMechanism.HV: 'pink',
            # EvaluationMechanism.C: 'red',
            # EvaluationMechanism.PO: 'green',
            # 1.0: 'red',
            # 0.9: 'fuchsia',
            # 0.8: 'orange',
            # 0.7: 'pink',
            # 0.6: 'yellow',
            0.5: 'green',
            # 0.4: 'cyan',
            # 0.3: 'blue'
        },
        # AgentType.SCALARIZED: {
        # EvaluationMechanism.SCALARIZED: 'cyan'
        # }
        # AgentType.PQL_EXP_3: {
        #     0.5: 'red',
        #     0.4: 'fuchsia',
        #     0.3: 'orange',
        #     0.2: 'pink',
        # }
    }

    graph_configurations = {
        # GraphType.EPISODES: {
        #     'limit': 100,
        #     'interval': 10
        # },
        GraphType.STEPS: {
            'limit': 20 * 1000,
            'interval': 200
        },
        # GraphType.MEMORY: {
        #     'limit': 200,
        #     'interval': 10
        # },
        # GraphType.VECTORS_PER_CELL: {
        #     'limit': 200,
        #     'interval': 10
        # },
        # GraphType.TIME: {
        #     'limit': 30,
        #     'interval': 2
        # },
    }

    ug.test_agents(environment=environment, hv_reference=hv_reference, epsilon=epsilon, alpha=alpha,
                   states_to_observe=states_to_observe, integer_mode=integer_mode, number_of_agents=number_of_agents,
                   agents_configuration=agents_configuration, gamma=gamma, max_steps=max_steps,
                   evaluation_mechanism=evaluation_mechanism, variable=variable,
                   graph_configuration=graph_configurations, solution=solution)


if __name__ == '__main__':
    main()
