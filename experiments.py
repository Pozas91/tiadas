import time

import utils.graphs as ug
import utils.miscellaneous as um
from agents import AgentA1, AgentPQL, AgentMOSP
from environments import Environment, DeepSeaTreasure
from models import GraphType, Vector, EvaluationMechanism, AgentType


def test_agents(env: Environment, hv_reference: Vector, variable: str, graph_types: dict,
                agents_configuration: dict, epsilon: float = 0.1, alpha: float = 1., max_steps: int = None,
                states_to_observe: list = None, episodes: int = 1000, integer_mode: bool = False,
                number_of_agents: int = 30, gamma: float = 1.,
                eval_mechanism: EvaluationMechanism = EvaluationMechanism.C):
    """
    This method run an experiment with the parameters and environment given
    :param variable:
    :param eval_mechanism:
    :param agents_configuration:
    :param integer_mode:
    :param env:
    :param hv_reference:
    :param epsilon:
    :param alpha:
    :param states_to_observe:
    :param episodes:
    :param graph_types:
    :param number_of_agents:
    :param gamma:
    :param max_steps:
    :return:
    """

    # Parameters
    if states_to_observe is None:
        states_to_observe = [env.initial_state]

    # Build environment
    env_name = env.__class__.__name__
    env_name_snake = um.str_to_snake_case(env_name)

    # File timestamp
    timestamp = int(time.time())

    # Extract tolerances
    relative_tolerance = Vector.relative_tolerance
    absolute_tolerance = Vector.absolute_tolerance

    ug.write_config_file(timestamp=timestamp, number_of_agents=number_of_agents, env_name_snake=env_name_snake,
                         seed=','.join(map(str, range(number_of_agents))), epsilon=epsilon, alpha=alpha,
                         relative_tolerance=relative_tolerance, max_steps=max_steps, variable=variable,
                         absolute_tolerance=absolute_tolerance, gamma=gamma, episodes=episodes)

    # Create graphs structure
    graphs = ug.initialize_graph_data(graph_types=graph_types, agents_configuration=agents_configuration)

    # Data max length
    data_max_len = float('-inf')

    # Information
    print('Environment: {}'.format(env_name))

    for i_agent in range(number_of_agents):

        print("\tExecution: {}".format(i_agent + 1))

        # Set a seed
        seed = i_agent

        for agent_type in agents_configuration:

            print('\t\tAgent: {}'.format(agent_type.value))

            for configuration in agents_configuration[agent_type].keys():

                print('\t\t\t{}: {}'.format(variable, configuration), end=' ')

                # Mark of time
                t0 = time.time()

                # Reset environment
                env.reset()
                env.seed(seed=seed)

                # Default values
                v_s_0 = None
                agent = None

                # Variable parameters
                parameters = {
                    'epsilon': epsilon, 'alpha': alpha, 'gamma': gamma, 'max_steps': max_steps,
                    'evaluation_mechanism': eval_mechanism
                }

                # Modify current configuration
                parameters.update({variable: configuration})

                # Is a SCALARIZED agent?
                if agent_type == AgentType.SCALARIZED:

                    # Removing useless parameters
                    del parameters['evaluation_mechanism']

                    # Set weights
                    weights = (.99, .01)

                    # Build agent
                    agent = AgentMOSP(seed=seed, environment=env, weights=weights,
                                      states_to_observe=states_to_observe, graph_types=set(graph_types.keys()),
                                      hv_reference=hv_reference, **parameters)

                    # Search one extreme objective
                    agent.train(episodes=episodes)

                    # Get p point from agent test
                    p = agent.get_accumulated_reward(from_state=states_to_observe[0])

                    # Add point found to pareto's frontier found
                    agent.pareto_frontier_found.append(p)

                    # Reset agent to train again with others weights
                    agent.reset()
                    agent.reset_totals()

                    # Set weights to find another extreme point
                    agent.weights = (.01, .99)

                    # Search the other extreme objective
                    agent.train(episodes=episodes)

                    # Get q point from agent test.
                    q = agent.get_accumulated_reward(from_state=states_to_observe[0])

                    # Add point found to pareto's frontier found
                    agent.pareto_frontier_found.append(q)

                    # Search pareto points
                    agent.calc_frontier_scalarized(p=p, q=q)

                    # Non-dominated vectors found in V(s0)
                    v_s_0 = agent.pareto_frontier_found

                # Is a PQL agent?
                elif agent_type == AgentType.PQL:

                    # Removing useless parameters
                    del parameters['alpha']

                    # Build an instance of agent
                    agent = AgentPQL(environment=env, seed=seed, hv_reference=hv_reference,
                                     graph_types=set(graph_types.keys()), states_to_observe=states_to_observe,
                                     integer_mode=integer_mode, **parameters)

                    # Train the agent
                    agent.train(episodes=episodes)

                    # Non-dominated vectors found in V(s0)
                    v_s_0 = agent.q_set_from_state(state=agent.environment.initial_state)

                # Is an A1 agent?
                elif agent_type == AgentType.A1:

                    # Build an instance of agent
                    agent = AgentA1(environment=env, seed=seed, hv_reference=hv_reference,
                                    graph_types=set(graph_types.keys()), states_to_observe=states_to_observe,
                                    integer_mode=integer_mode, **parameters)

                    # Train the agent
                    agent.train(episodes=episodes)

                    # Non-dominated vectors found in V(s0)
                    v_real = agent.v_real()

                    v_s_0 = v_real.get(agent.environment.initial_state, {
                        0: agent.environment.default_reward.zero_vector
                    }).values()

                else:
                    ValueError("Agent type does not valid!")

                print('-> {:.2f}s'.format(time.time() - t0))

                # Write vectors found into file
                ug.write_v_from_initial_state_file(timestamp=timestamp, seed=i_agent, env_name_snake=env_name_snake,
                                                   v_s_0=v_s_0, variable=variable, agent_type=agent_type,
                                                   configuration=configuration)

                # Calc data maximum length
                data_max_len = ug.update_graphs(agent=agent, graphs=graphs,
                                                configuration=str(configuration), states_to_observe=states_to_observe,
                                                agent_type=agent_type)

    ug.prepare_data_and_show_graph(timestamp=timestamp, env_name=env_name, env_name_snake=env_name_snake, graphs=graphs,
                                   number_of_agents=number_of_agents, agents_configuration=agents_configuration,
                                   alpha=alpha, epsilon=epsilon, gamma=gamma, stop_condition={'episodes': episodes},
                                   max_steps=max_steps, initial_state=env.initial_state, integer_mode=integer_mode,
                                   variable=variable)


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
    execution_time = 30
    steps_limit = 10000

    # Environment configuration
    # environment = PressurizedBountifulSeaTreasure(initial_state=initial_state)
    # hv_reference = Vector([-25, 0, -120])
    environment = DeepSeaTreasure(initial_state=initial_state)
    hv_reference = Vector([-25, 0])
    solution = environment.pareto_optimal

    # Variable parameters
    variable = 'alpha'
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
            EvaluationMechanism.HV: 'pink',
            EvaluationMechanism.C: 'red',
            # EvaluationMechanism.PO: 'green'
        },
        # AgentType.SCALARIZED: {
        # EvaluationMechanism.SCALARIZED: 'cyan'
        # }
    }

    graph_types = {
        GraphType.STEPS,
        GraphType.MEMORY,
        GraphType.VECTORS_PER_CELL,
        GraphType.TIME,
        GraphType.EPISODES
    }

    ug.test_time(environment=environment, hv_reference=hv_reference, epsilon=epsilon, alpha=alpha,
                 states_to_observe=states_to_observe, integer_mode=integer_mode, number_of_agents=number_of_agents,
                 agents_configuration=agents_configuration, gamma=gamma, max_steps=max_steps,
                 evaluation_mechanism=evaluation_mechanism, variable=variable, execution_time=execution_time,
                 seconds_to_get_data=1, solution=solution)

    # ug.test_steps(environment=environment, hv_reference=hv_reference, epsilon=epsilon, alpha=alpha,
    #               states_to_observe=states_to_observe, integer_mode=integer_mode, number_of_agents=number_of_agents,
    #               agents_configuration=agents_configuration, gamma=gamma, max_steps=max_steps,
    #               evaluation_mechanism=evaluation_mechanism, variable=variable, steps_limit=steps_limit,
    #               steps_to_get_data=1)

    # Vector.decimals_allowed = decimals_allowed

    # for tolerance in [0.1, 0.3, 0.5]:
    #     Vector.set_absolute_tolerance(absolute_tolerance=tolerance, integer_mode=True)
    #
    #     test_agents(env=PressurizedBountifulSeaTreasure(initial_state=initial_state),
    #                 hv_reference=Vector([-25, 0, -120]), epsilon=0.7, alpha=alpha, states_to_observe=[initial_state],
    #                 episodes=episodes, integer_mode=True, graph_types=graph_types, number_of_agents=number_of_agents,
    #                 agents_configuration=agents_configuration, gamma=gamma, max_steps=max_steps,
    #                 eval_mechanism=evaluation_mechanism, variable=variable)

    # test_agents(environment=DeepSeaTreasureRightDown(initial_state=initial_state, columns=columns),
    #             hv_reference=Vector([-25, 0]), epsilon=0.7, alpha=alpha, states_to_observe=[initial_state],
    #             episodes=episodes, integer_mode=True, graph_types=graph_types, number_of_agents=number_of_agents,
    #             agents_configuration=agents_configuration, gamma=gamma, max_steps=max_steps,
    #             evaluation_mechanism=evaluation_mechanism, variable=variable)

    # test_agents(environment=MoPuddleWorldAcyclic(), hv_reference=Vector([-50, -150]), epsilon=0.3, alpha=alpha,
    #             states_to_observe=[(2, 8)], episodes=episodes, integer_mode=True, graph_types=graph_types,
    #             number_of_agents=number_of_agents, agents_configuration=agents_configuration, gamma=gamma,
    #             max_steps=max_steps)
    #
    # test_agents(environment=SpaceExplorationAcyclic(), hv_reference=Vector([-150, -150]), epsilon=0.3, alpha=alpha,
    #             states_to_observe=[(0, 0)], episodes=episodes, integer_mode=True, graph_types=graph_types,
    #             number_of_agents=number_of_agents, agents_configuration=agents_configuration, gamma=gamma,
    #             max_steps=max_steps)

    # agents_configuration = {**agent_a1_configuration, **agent_scalarized_configuration}
    #
    # test_agents(environment=DeepSeaTreasureRightDownStochastic(), hv_reference=Vector([-25, 0]), epsilon=0.7,
    #             alpha=alpha, states_to_observe=[(0, 0)], episodes=episodes, integer_mode=True, graph_types=graph_types,
    #             number_of_agents=number_of_agents, agents_configuration=agents_configuration, gamma=gamma,
    #             max_steps=max_steps)
    #
    # agents_configuration = {**agent_a1_configuration, **agent_pql_configuration}
    #
    # test_agents(environment=BonusWorldAcyclic(), hv_reference=Vector([-50, -50, -50]), epsilon=0.25, alpha=alpha,
    #             states_to_observe=[(0, 0)], episodes=episodes, integer_mode=True, graph_types=graph_types,
    #             number_of_agents=number_of_agents, agents_configuration=agents_configuration, gamma=gamma,
    #             max_steps=max_steps)
    #
    # agents_configuration = {**agent_a1_configuration}
    #
    # test_agents(environment=PressurizedBountifulSeaTreasureRightDownStochastic(), hv_reference=Vector([-25, 0, -100]),
    #             epsilon=0.7, alpha=alpha, states_to_observe=[(0, 0)], episodes=episodes, integer_mode=True,
    #             graph_types=graph_types, number_of_agents=number_of_agents, agents_configuration=agents_configuration,
    #             gamma=gamma, max_steps=max_steps)


if __name__ == '__main__':
    main()
