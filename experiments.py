from colorama import init

import utils.graphs as ug
from environments import DeepSeaTreasureStochastic, DeepSeaTreasure, DeepSeaTreasureRightDown, \
    DeepSeaTreasureRightDownStochastic
from models import GraphType, AgentType, Vector

init(autoreset=True)


def main():
    # Basic configuration
    alpha = None
    number_of_agents = 2
    gamma = 1.
    max_steps = None
    initial_state = (0, 0)
    evaluation_mechanism = None
    epsilon = None
    states_to_observe = [initial_state]
    integer_mode = False
    solution = None

    # Environment configuration
    # environment = DeepSeaTreasureRightDown(initial_state=initial_state)
    environment = DeepSeaTreasureRightDownStochastic(initial_state=initial_state)
    # environment = DeepSeaTreasure(initial_state=initial_state)
    # environment = SpaceExploration(initial_state=initial_state)
    # environment = DeepSeaTreasureStochastic(initial_state=initial_state)
    hv_reference = environment.hv_reference

    # Variable parameters
    variable = 'gamma'

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
            1.0: 'red',
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
        #     'limit': 20 * 1000,
        #     'interval': 200
        # },
        GraphType.SWEEP: {
            'limit': 25,
            'interval': 1
        },
        GraphType.MEMORY: {
            'limit': 25,
            'interval': 1
        },
        # GraphType.DATA_PER_STATE: {
        #     'limit': 30,
        #     # 'interval': 2
        # },
        # GraphType.TIME: {
        #     'limit': 10,
        #     'interval': 2
        # },
    }

    for decimals_allowed in [1, 0, 2, 3]:

        # Decimals allowed
        Vector.set_decimals_allowed(decimals_allowed=decimals_allowed)

        ug.test_agents(environment=environment, hv_reference=hv_reference, epsilon=epsilon, alpha=alpha,
                       states_to_observe=states_to_observe, integer_mode=integer_mode,
                       number_of_agents=number_of_agents, agents_configuration=agents_configuration, gamma=gamma,
                       max_steps=max_steps, evaluation_mechanism=evaluation_mechanism, variable=variable,
                       graph_configuration=graph_configurations, solution=solution)

    # for evaluation_mechanism in [
    #     EvaluationMechanism.HV, EvaluationMechanism.CHV, EvaluationMechanism.C, EvaluationMechanism.PO
    # ]:
    #     ug.test_agents(environment=environment, hv_reference=hv_reference, epsilon=epsilon, alpha=alpha,
    #                    states_to_observe=states_to_observe, integer_mode=integer_mode,
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
    #                    states_to_observe=states_to_observe, integer_mode=integer_mode,
    #                    number_of_agents=number_of_agents,
    #                    agents_configuration=agents_configuration, gamma=gamma, max_steps=max_steps,
    #                    evaluation_mechanism=evaluation_mechanism, variable=variable,
    #                    graph_configuration=graph_configuration, solution=solution)


if __name__ == '__main__':
    main()
