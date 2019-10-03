import utils.graphs as ug
from environments import DeepSeaTreasure
from models import GraphType, Vector, EvaluationMechanism, AgentType


def main():
    # Basic configuration
    alpha = 0.8
    number_of_agents = 5
    episodes = 2000
    gamma = 1.
    max_steps = 250
    initial_state = (0, 0)
    columns = 10
    evaluation_mechanism = EvaluationMechanism.C
    decimals_allowed = 7
    epsilon = 0.7
    states_to_observe = [initial_state]
    integer_mode = False
    execution_time = 60
    steps_limit = 450000

    # Environment configuration
    # environment = PressurizedBountifulSeaTreasure(initial_state=initial_state)
    # hv_reference = Vector([-25, 0, -120])
    environment = DeepSeaTreasure(initial_state=initial_state)
    hv_reference = Vector([-25, 0])
    solution = environment.pareto_optimal

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
            0.7: 'pink',
            # 0.6: 'yellow',
            # 0.5: 'green',
            # 0.4: 'cyan',
            # 0.3: 'blue'
        },
        # AgentType.SCALARIZED: {
        # EvaluationMechanism.SCALARIZED: 'cyan'
        # }
    }

    graph_configurations = {
        # GraphType.EPISODES: {
        #     'limit': 1000,
        #     'interval': 10
        # },
        # GraphType.STEPS: {
        #     'limit': 4000,
        #     'interval': 10
        # },
        GraphType.MEMORY: {
            GraphType.EPISODES: {
                'limit': 1000,
                'interval': 10
            }
        },
        # GraphType.VECTORS_PER_CELL: {
        #
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
