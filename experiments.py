from itertools import permutations

from colorama import init

import utils.graphs as ug
from environments import SpaceExploration
from models import GraphType, EvaluationMechanism, AgentType

init(autoreset=True)


def main():
    # Basic configuration
    alpha = 1
    number_of_agents = 2
    gamma = 1.
    max_steps = 250
    initial_state = (5, 2)
    evaluation_mechanism = EvaluationMechanism.HV
    decimals_allowed = 7
    epsilon = 0.7
    states_to_observe = [initial_state]
    integer_mode = False
    solution = None

    # Environment configuration
    # environment = PressurizedBountifulSeaTreasure(initial_state=initial_state)
    # hv_reference = Vector([-25, 0, -120])
    # environment = DeepSeaTreasure(initial_state=initial_state)
    # hv_reference = Vector([-25, 0])
    # hv_reference = Vector([0, 0, -150])
    environment = SpaceExploration(initial_state=initial_state)
    hv_reference = environment.hv_reference

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
            1.0: 'red',
            # 0.9: 'fuchsia',
            # 0.8: 'orange',
            # 0.7: 'pink',
            0.6: 'yellow',
            # 0.5: 'green',
            # 0.4: 'cyan',
            0.3: 'blue'
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
        #     'limit': 500,
        #     'interval': 5
        # },
        # GraphType.STEPS: {
        #     'limit': 1000,
        #     'interval': 20
        # },
        # GraphType.MEMORY: {
        #     'limit': 200,
        #     'interval': 5
        # },
        GraphType.VECTORS_PER_CELL: {
            'limit': 300,
            # 'interval': 2
        },
        # GraphType.TIME: {
        #     'limit': 10,
        #     'interval': 2
        # },
    }

    ug.test_agents(environment=environment, hv_reference=hv_reference, epsilon=epsilon, alpha=alpha,
                   states_to_observe=states_to_observe, integer_mode=integer_mode,
                   number_of_agents=number_of_agents, agents_configuration=agents_configuration, gamma=gamma,
                   max_steps=max_steps, evaluation_mechanism=evaluation_mechanism, variable=variable,
                   graph_configuration=graph_configurations, solution=solution)

    # permutations_graph = permutations(graph_configurations.keys(), 2)
    #
    # for permutation in permutations_graph:
    #     # Extract information of this permutation
    #     graph_configuration = {element: graph_configurations[element] for element in permutation}
    #
    #     ug.test_agents(environment=environment, hv_reference=hv_reference, epsilon=epsilon, alpha=alpha,
    #                    states_to_observe=states_to_observe, integer_mode=integer_mode,
    #                    number_of_agents=number_of_agents,
    #                    agents_configuration=agents_configuration, gamma=gamma, max_steps=max_steps,
    #                    evaluation_mechanism=evaluation_mechanism, variable=variable,
    #                    graph_configuration=graph_configuration, solution=solution)


if __name__ == '__main__':
    main()
