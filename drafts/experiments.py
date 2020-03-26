import utils.graphs as ug
from environments import PyramidMDP
from models import AgentType, GraphType, EvaluationMechanism


def main():
    # Basic configuration
    alpha = 0.1
    number_of_agents = 1
    gamma = 1.
    max_steps = 250
    initial_state = (0, 0)
    evaluation_mechanism = EvaluationMechanism.PO
    epsilon = 0.7
    states_to_observe = [initial_state]

    # Environment configuration
    # diagonals = 4
    # environment = DeepSeaTreasureRightDown(initial_state=initial_state)
    # environment = DeepSeaTreasureRightDownStochastic(initial_state=initial_state, diagonals=diagonals)
    # environment = DeepSeaTreasure(initial_state=initial_state)
    # environment = SpaceExploration(initial_state=initial_state)
    # environment = DeepSeaTreasureStochastic(initial_state=initial_state)
    # hv_reference = environment.hv_reference
    # solution = environment.pareto_optimal[:diagonals]

    # Advanced configuration

    # Variable parameters
    # variable = 'gamma'
    # variable = 'alpha'
    variable = 'decimal_precision'

    agents_configuration = {
        # AgentType.A1: {
        #     # EvaluationMechanism.HV: 'yellow',
        #     # EvaluationMechanism.C:  'orange',
        #     # EvaluationMechanism.PO: 'blue',
        #     # 1.0: 'red',
        #     # 0.9: 'fuchsia',
        #     # 0.8: 'orange',
        #     # 0.7: 'pink',
        #     # 0.6: 'yellow',
        #     # 0.5: 'green',
        #     # 0.4: 'cyan',
        #     # 0.3: 'blue'
        #
        #     # Decimals allowed
        #     # 1: 'red',
        #     # 2: 'fuchsia',
        #     3: 'orange',
        # },
        # AgentType.PQL: {
        #     # EvaluationMechanism.HV: 'pink',
        #     # EvaluationMechanism.C: 'red',
        #     # EvaluationMechanism.PO: 'green',
        #     1.0: 'red',
        #     # 0.9: 'fuchsia',
        #     0.8: 'orange',
        #     0.7: 'pink',
        #     # 0.6: 'yellow',
        #     0.5: 'green',
        #     # 0.4: 'cyan',
        #     0.3: 'blue'
        # },
        # AgentType.SCALARIZED: {
        # EvaluationMechanism.SCALARIZED: 'cyan'
        # }
        # AgentType.PQL_EXP_3: {
        #     0.5: 'red',
        #     0.4: 'fuchsia',
        #     0.3: 'orange',
        #     0.2: 'pink',
        # },
        AgentType.W: {
            1: 'red',
            2: 'fuchsia',
            3: 'orange',
            # 4: 'pink',
            # 10: 'yellow'
            # 1.0: 'red',
            # 0.9: 'fuchsia',
            # 0.8: 'orange',
            # 0.7: 'pink',
            # 0.6: 'yellow',
            # 0.5: 'green',
            # 0.4: 'cyan',
            # 0.3: 'blue'
        }
    }

    graph_configurations = {
        # GraphType.EPISODES: {
        #     'limit': 500,
        #     'interval': 5
        # },
        # GraphType.STEPS: {
        #     'limit': 700 * 1000,
        #     'interval': 200
        # },
        GraphType.SWEEP: {
            'limit': 5,
            'interval': 1
        },
        # GraphType.V_S_0: {
        #     'limit': 20,
        #     'interval': 1
        # },
        GraphType.MEMORY: {
            'limit': 5,
            'interval': 1
        },
        # GraphType.DATA_PER_STATE: {
        #     'limit': 30,
        #     # 'interval': 2
        # },
        GraphType.TIME: {
            'limit': 5,
            'interval': 2
        },
    }

    # Vector.set_decimal_precision(decimal_precision=0.01)

    # for diagonals in range(10):
    #     diagonals += 1

    columns = 2

    # environment = DeepSeaTreasureRightDown(initial_state=initial_state)
    # environment = DeepSeaTreasureRightDownStochastic(initial_state=initial_state, diagonals=diagonals)
    environment = PyramidMDP(diagonals=1)
    # environment = DeepSeaTreasure(initial_state=initial_state)
    # environment = SpaceExploration(initial_state=initial_state)
    # environment = DeepSeaTreasureStochastic(initial_state=initial_state)

    hv_reference = environment.hv_reference
    solution = environment.pareto_optimal[:columns]

    ug.test_agents(environment=environment, hv_reference=hv_reference, epsilon=epsilon, alpha=alpha,
                   states_to_observe=states_to_observe, number_of_agents=number_of_agents,
                   agents_configuration=agents_configuration, gamma=gamma, max_steps=max_steps,
                   evaluation_mechanism=evaluation_mechanism, variable=variable,
                   graph_configuration=graph_configurations, solution=solution)

    # for evaluation_mechanism in [
    #     EvaluationMechanism.HV, EvaluationMechanism.CHV, EvaluationMechanism.C, EvaluationMechanism.PO
    # ]:
    #     ug.test_agents(environment=environment, hv_reference=hv_reference, epsilon=epsilon, alpha=alpha,
    #                    states_to_observe=states_to_observe,
    #                    number_of_agents=number_of_agents, agents_configuration=agents_configuration, gamma=gamma,
    #                    max_steps=max_steps, evaluation_mechanism=evaluation_mechanism, variable=variable,
    #                    graph_configuration=graph_configurations, solution=solution)

    # permutations_graph = permutations(graph_configurations.keys(), 2)
    #
    # for permutation in permutations_graph:
    #     # Extract information of this permutation
    #     graph_configuration = {element: graph_configurations[element] for element in permutation}
    #
    #     ug.test_agents(environment=environment, hv_reference=hv_reference, epsilon=epsilon, alpha=alpha,
    #                    states_to_observe=states_to_observe,
    #                    number_of_agents=number_of_agents,
    #                    agents_configuration=agents_configuration, gamma=gamma, max_steps=max_steps,
    #                    evaluation_mechanism=evaluation_mechanism, variable=variable,
    #                    graph_configuration=graph_configuration, solution=solution)


if __name__ == '__main__':
    main()
