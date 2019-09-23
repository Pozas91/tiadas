import math
import time
from enum import Enum
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import utils.miscellaneous as um
from agents import Agent, AgentPQL, AgentMOSP, AgentA1
from environments import Environment, DeepSeaTreasureRightDown
from models import Vector, EvaluationMechanism, GraphType


class AgentType(Enum):
    A1 = 'a1'
    PQL = 'pql'
    SCALARIZED = 'scalarized'


def write_config_file(timestamp: int, number_of_agents: int, env_name_snake: str, **kwargs):
    # Path to save file
    config_path = './dumps/config/{}_{}_{}.config'

    filename_config = Path(
        config_path.format(timestamp, number_of_agents, env_name_snake).lower()
    )

    with filename_config.open(mode='w+') as file:
        file_data = ''

        for key, value in kwargs.items():
            file_data += '{} = {}\n'.format(key, value)

        file.write(file_data)


def write_v_from_initial_state_file(timestamp: int, i_agent: int, env_name_snake: str, v_s_0: list,
                                    agent_type: AgentType, variable, configuration):
    """
    Write V(s0) data.
    :param configuration:
    :param variable:
    :param timestamp:
    :param i_agent:
    :param env_name_snake:
    :param v_s_0:
    :param agent_type:
    :return:
    """
    # Path to save file
    v_s_0_path = './dumps/{}/data/{}_{}_{}_{}_{}.yml'

    # Order vectors by origin Vec(0) nearest
    v_s_0 = um.order_vectors_by_origin_nearest(vectors=v_s_0)

    v_s_0_filename = Path(
        v_s_0_path.format(str(agent_type.value), timestamp, i_agent, env_name_snake, variable, configuration).lower()
    )

    with v_s_0_filename.open(mode='w+') as file:
        file_data = 'v_s_0_non_dominated:\n'
        file_data += '  {}'.format('\n  '.join(map(str, v_s_0)))

        file.write(file_data)


def initialize_graph_data(graph_types: dict, agents_configuration: dict) -> dict:
    """
    Initialize graph data dictionary

    :param graph_types:
    :param agents_configuration:
    :return:
    """

    # Create graphs structure
    graphs = dict()

    for graph_type in graph_types.keys():
        data_types = dict()

        for agent_type in agents_configuration:
            data_evaluations = dict()

            for configuration in agents_configuration[agent_type].keys():
                data_evaluations.update({
                    '{}'.format(configuration): list()
                })

            data_types.update({
                agent_type: data_evaluations
            })

        graphs.update({
            graph_type: data_types
        })

    return graphs


def test_agents(environment: Environment, hv_reference: Vector, variable: str, graph_types: dict,
                agents_configuration: dict, epsilon: float = 0.1, alpha: float = 1., max_steps: int = None,
                states_to_observe: list = None, epochs: int = 1000, integer_mode: bool = False,
                number_of_agents: int = 30, gamma: float = 1.,
                evaluation_mechanism: EvaluationMechanism = EvaluationMechanism.C):
    """
    This method run an experiment with the parameters and environment given
    :param variable:
    :param evaluation_mechanism:
    :param agents_configuration:
    :param integer_mode:
    :param environment:
    :param hv_reference:
    :param epsilon:
    :param alpha:
    :param states_to_observe:
    :param epochs:
    :param graph_types:
    :param number_of_agents:
    :param gamma:
    :param max_steps:
    :return:
    """

    # Parameters
    if states_to_observe is None:
        states_to_observe = [environment.initial_state]

    # Build environment
    env_name = environment.__class__.__name__
    env_name_snake = um.str_to_snake_case(env_name)

    # File timestamp
    timestamp = int(time.time())

    # Extract tolerances
    relative_tolerance = Vector.relative_tolerance
    absolute_tolerance = Vector.absolute_tolerance

    write_config_file(timestamp=timestamp, number_of_agents=number_of_agents, env_name_snake=env_name_snake,
                      seed=','.join(map(str, range(number_of_agents))), epsilon=epsilon, alpha=alpha,
                      relative_tolerance=relative_tolerance, max_steps=max_steps, variable=variable,
                      absolute_tolerance=absolute_tolerance, gamma=gamma, epochs=epochs)

    # Create graphs structure
    graphs = initialize_graph_data(graph_types=graph_types, agents_configuration=agents_configuration)

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
                environment.reset()
                environment.seed(seed=seed)

                # Default values
                v_s_0 = None
                agent = None

                # Variable parameters
                parameters = {
                    'epsilon':              epsilon, 'alpha': alpha, 'gamma': gamma, 'max_steps': max_steps,
                    'evaluation_mechanism': evaluation_mechanism
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
                    agent = AgentMOSP(seed=seed, environment=environment, weights=weights,
                                      states_to_observe=states_to_observe, graph_types=set(graph_types.keys()),
                                      hv_reference=hv_reference, **parameters)

                    # Search one extreme objective
                    agent.train(epochs=epochs)

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
                    agent.train(epochs=epochs)

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
                    agent = AgentPQL(environment=environment, seed=seed, hv_reference=hv_reference,
                                     graph_types=set(graph_types.keys()), states_to_observe=states_to_observe,
                                     integer_mode=integer_mode, **parameters)

                    # Train the agent
                    agent.train(epochs=epochs)

                    # Non-dominated vectors found in V(s0)
                    v_s_0 = agent.non_dominated_vectors_from_state(state=agent.environment.initial_state)

                # Is an A1 agent?
                elif agent_type == AgentType.A1:

                    # Build an instance of agent
                    agent = AgentA1(environment=environment, seed=seed, hv_reference=hv_reference,
                                    graph_types=set(graph_types.keys()), states_to_observe=states_to_observe,
                                    integer_mode=integer_mode, **parameters)

                    # Train the agent
                    agent.train(epochs=epochs)

                    # Non-dominated vectors found in V(s0)
                    v_real = agent.v_real()

                    v_s_0 = v_real.get(agent.environment.initial_state, {
                        0: agent.environment.default_reward.zero_vector
                    }).values()

                else:
                    ValueError("Agent type does not valid!")

                print('-> {:.2f}s'.format(time.time() - t0))

                # Write vectors found into file
                write_v_from_initial_state_file(timestamp=timestamp, i_agent=i_agent, env_name_snake=env_name_snake,
                                                v_s_0=v_s_0, variable=variable, agent_type=agent_type,
                                                configuration=configuration)

                # Calc data maximum length
                data_max_len = update_graph(agent=agent, data_max_len=data_max_len, graphs=graphs,
                                            configuration=str(configuration), states_to_observe=states_to_observe,
                                            agent_type=agent_type)

    prepare_data_and_show_graph(timestamp=timestamp, data_max_len=data_max_len, env_name=env_name,
                                env_name_snake=env_name_snake, graphs=graphs, number_of_agents=number_of_agents,
                                agents_configuration=agents_configuration, alpha=alpha, epsilon=epsilon, gamma=gamma,
                                epochs=epochs, max_steps=max_steps, initial_state=environment.initial_state,
                                integer_mode=integer_mode, variable=variable, graph_types=graph_types)


def update_graph(agent: Agent, data_max_len: int, configuration: str, graphs: dict, states_to_observe: list,
                 agent_type: AgentType):
    """
    Update graph to show
    :param configuration:
    :param agent_type:
    :param agent:
    :param data_max_len:
    :param graphs:
    :param states_to_observe:
    :return:
    """

    for graph_type in graphs:
        # Recover old data
        data = graphs[graph_type][agent_type][configuration]

        if graph_type is GraphType.MEMORY:
            agent_data = agent.graph_info[graph_type]
        elif graph_type is GraphType.VECTORS_PER_CELL:
            agent_data = agent.graph_info[graph_type]
        else:
            # Prepare new data
            agent_data = agent.graph_info[graph_type][states_to_observe[0]]

        data.append(agent_data)

        # Update data in the dictionary
        graphs[graph_type][agent_type].update({
            configuration: data
        })

        if graph_type is not GraphType.VECTORS_PER_CELL:
            data_max_len = max(data_max_len, len(agent_data))

    return data_max_len


def prepare_data_and_show_graph(timestamp: int, data_max_len: int, env_name: str, env_name_snake: str, graphs: dict,
                                number_of_agents: int, agents_configuration: dict, alpha: float, gamma: float,
                                epsilon: float, epochs: int, max_steps: int, initial_state: tuple, integer_mode: bool,
                                variable: str, graph_types: dict):
    """
    Prepare data to show a graph with the information about results
    :param graph_types:
    :param variable:
    :param integer_mode:
    :param initial_state:
    :param max_steps:
    :param epochs:
    :param alpha:
    :param gamma:
    :param epsilon:
    :param agents_configuration:
    :param timestamp:
    :param data_max_len:
    :param env_name:
    :param env_name_snake:
    :param graphs:
    :param number_of_agents:
    :return:
    """

    # Path to save file
    graph_path = './dumps/{}/graphs/{}_{}_{}_{}_{}_{}.m'
    plot_path = './dumps/plots/{}_{}_{}.png'

    # Extract vectors per cells graph from all graphs
    vectors_per_cells_graph = graphs.pop(GraphType.VECTORS_PER_CELL, {})

    for agent_type in vectors_per_cells_graph:

        for configuration in agents_configuration[agent_type].keys():
            # Recover old data
            data = vectors_per_cells_graph[agent_type][str(configuration)]
            process_data = np.average(data, axis=0)

            filename_m = Path(
                graph_path.format(agent_type.value, timestamp, number_of_agents, env_name_snake,
                                  GraphType.VECTORS_PER_CELL.value, variable, str(configuration)).lower()
            )

            with filename_m.open(mode='w+') as file:
                file_data = "Z = [\n{}\n];\n".format(''.join([';\n'.join(map(str, x)) for x in process_data]))
                # file_data += "means = mean(Y);\n"
                file_data += 'figure;\n'
                file_data += 'bar3(Z);\n'
                file_data += 'zlim([0, 30]);\n'
                file_data += "xlabel('Columns');\n"
                file_data += "ylabel('Rows');\n"
                file_data += "zlabel('# Vectors');\n"
                file_data += "title('{} {}');\n".format(variable, configuration)
                file.write(file_data)

    # Parameters
    parameters = {'sharex': True}

    # Graph instances
    fig, axs = plt.subplots(nrows=len(graphs), **parameters)

    # Resolution 1440x1080 (4:3)
    fig.set_size_inches(14.4, 10.8)
    fig.set_dpi(244)

    for axs_i, graph_type in enumerate(graphs):

        graph_name = graph_type.value

        for agent_type in graphs[graph_type]:

            for configuration in agents_configuration[agent_type].keys():

                # Recover old data
                data = graphs[graph_type][agent_type][str(configuration)]
                color = agents_configuration[agent_type][configuration]

                # Change to same length at all arrays
                for i, x_steps in enumerate(data):

                    # Length of x
                    len_x = len(x_steps)

                    # Difference
                    difference = data_max_len - len_x

                    # If x is not empty
                    if len_x > 0:
                        data[i] = np.append(x_steps, [x_steps[-1]] * difference)
                    else:
                        data[i] = [0] * difference

                process_data = np.average(data, axis=0)
                error = np.std(data, axis=0)
                x = np.arange(0, data_max_len, 1)

                # Set graph to current evaluation mechanism
                axs[axs_i].errorbar(x=x, y=process_data, yerr=error, errorevery=math.ceil(data_max_len * 0.1),
                                    label='{} {}'.format(agent_type.value, configuration), color=color)

                filename_m = Path(
                    graph_path.format(agent_type.value, timestamp, number_of_agents, env_name_snake, graph_name,
                                      variable, str(configuration)).lower()
                )

                with filename_m.open(mode='w+') as file:
                    file_data = "x = [{}]\n".format(', '.join(map(str, x)))
                    file_data += "Y = [\n{}\n]\n".format(';\n'.join([', '.join(map(str, x)) for x in data]))
                    file_data += "means = mean(Y);\n"
                    file_data += 'figure;\n'
                    file_data += 'plot(x, means);\n'
                    file.write(file_data)

            # Show data
            if graph_type == GraphType.STEPS:
                axs[axs_i].set_xlabel('{} x{}'.format(graph_type.value, AgentPQL.steps_to_get_graph_data))
            elif graph_type == GraphType.MEMORY:
                axs[axs_i].set_xlabel('{} (x{} steps)'.format(graph_type.value, AgentPQL.steps_to_get_graph_data))
            elif graph_type == GraphType.TIME:
                axs[axs_i].set_xlabel('{} x{}s'.format(graph_type.value, AgentPQL.seconds_to_get_graph_data))
            else:
                axs[axs_i].set_xlabel(graph_type.value)

            if graph_type == GraphType.MEMORY:
                axs[axs_i].set_ylabel('# of vectors')
            else:
                axs[axs_i].set_ylabel('HV max')

        # Shrink current axis by 20%
        box = axs[axs_i].get_position()

        axs[axs_i].set_position([
            box.x0, box.y0, box.width * 0.9, box.height
        ])

        limits = graph_types[graph_type]['limits']
        limits_y = limits.get('y', False)
        limits_x = limits.get('x', False)

        if limits_y:
            axs[axs_i].set_ylim(limits_y)

        if limits_x:
            axs[axs_i].set_ylim(limits_x)

    fig.suptitle('{} environment'.format(env_name))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Text information
    props = {
        'boxstyle':  'round',
        'facecolor': 'wheat',
        'alpha':     0.5
    }

    multiply_factor = (10 ** Vector.decimals_allowed) if integer_mode else 1
    relative_tolerance = Vector.relative_tolerance
    absolute_tolerance = Vector.absolute_tolerance / multiply_factor

    basic_information = list()

    if variable is not 'alpha':
        basic_information.append(r'$\alpha={}$'.format(alpha))

    if variable is not 'gamma':
        basic_information.append(r'$\gamma={}$'.format(gamma))

    if variable is not 'epsilon':
        basic_information.append(r'$\epsilon={}$'.format(epsilon))

    if variable is not 'max_steps':
        basic_information.append(r'$max\_steps={}$'.format(max_steps))

    text_information = '\n'.join(basic_information + [
        r'$epochs={}$'.format(epochs),
        r'$initial\_state={}$'.format(initial_state),
        r'$relative\_tolerance={}$'.format(relative_tolerance),
        r'$absolute\_tolerance={}$'.format(absolute_tolerance),
        r'$\# agents={}$'.format(number_of_agents)
    ])

    plt.text(0.85, 0.5, text_information, bbox=props, transform=plt.gcf().transFigure)

    # Define figure path
    plot_filename = Path(
        plot_path.format(timestamp, number_of_agents, env_name_snake).lower()
    )

    # Save figure
    plt.savefig(plot_filename)

    plt.show()


def main():
    # Default parameters
    alpha = 0.1
    number_of_agents = 1
    epochs = 10
    gamma = 1.
    max_steps = 250
    initial_state = (0, 0)
    columns = 10
    evaluation_mechanism = EvaluationMechanism.C
    decimals_allowed = 7

    # Variable parameters
    # variable = 'alpha'
    variable = 'evaluation_mechanism'

    agents_configuration = {
        AgentType.A1: {
            EvaluationMechanism.HV: 'yellow',
            EvaluationMechanism.C:  'orange',
            EvaluationMechanism.PO: 'blue',
            # 0.01: 'blue',
            # 0.1: 'beige',
            # 0.3: 'gold',
            # 0.6: 'orange',
            # 0.8: 'fuchsia',
            # 1.0: 'cyan'
        },
        # AgentType.PQL:        {
        #     EvaluationMechanism.HV: 'pink',
        #     EvaluationMechanism.C:  'red',
        #     EvaluationMechanism.PO: 'green'
        # },
        # AgentType.SCALARIZED: {
        # EvaluationMechanism.SCALARIZED: 'cyan'
        # }
    }

    graph_types = {
        GraphType.STEPS:            {
            'limits': {
                'y': [0, 2000]
            }
        },
        # GraphType.MEMORY:           {
        #     'limits': {
        #         'y': [0, 700]
        #     }
        # },
        # GraphType.VECTORS_PER_CELL: {
        # }
        GraphType.TIME: {
            'limits': {
                'y': [0, 2000]
            }
        },
        # GraphType.EPOCHS: {
        # }
    }

    Vector.decimals_allowed = decimals_allowed

    for tolerance in [0.1, 0.3, 0.5]:
        Vector.set_absolute_tolerance(absolute_tolerance=tolerance, integer_mode=True)

        test_agents(environment=DeepSeaTreasureRightDown(initial_state=initial_state, columns=columns),
                    hv_reference=Vector([-25, 0]), epsilon=0.7, alpha=alpha, states_to_observe=[initial_state],
                    epochs=epochs, integer_mode=True, graph_types=graph_types, number_of_agents=number_of_agents,
                    agents_configuration=agents_configuration, gamma=gamma, max_steps=max_steps,
                    evaluation_mechanism=evaluation_mechanism, variable=variable)

    # test_agents(environment=DeepSeaTreasureRightDown(initial_state=initial_state, columns=columns),
    #             hv_reference=Vector([-25, 0]), epsilon=0.7, alpha=alpha, states_to_observe=[initial_state],
    #             epochs=epochs, integer_mode=True, graph_types=graph_types, number_of_agents=number_of_agents,
    #             agents_configuration=agents_configuration, gamma=gamma, max_steps=max_steps,
    #             evaluation_mechanism=evaluation_mechanism, variable=variable)

    # test_agents(environment=MoPuddleWorldAcyclic(), hv_reference=Vector([-50, -150]), epsilon=0.3, alpha=alpha,
    #             states_to_observe=[(2, 8)], epochs=epochs, integer_mode=True, graph_types=graph_types,
    #             number_of_agents=number_of_agents, agents_configuration=agents_configuration, gamma=gamma,
    #             max_steps=max_steps)
    #
    # test_agents(environment=SpaceExplorationAcyclic(), hv_reference=Vector([-150, -150]), epsilon=0.3, alpha=alpha,
    #             states_to_observe=[(0, 0)], epochs=epochs, integer_mode=True, graph_types=graph_types,
    #             number_of_agents=number_of_agents, agents_configuration=agents_configuration, gamma=gamma,
    #             max_steps=max_steps)

    # agents_configuration = {**agent_a1_configuration, **agent_scalarized_configuration}
    #
    # test_agents(environment=DeepSeaTreasureRightDownStochastic(), hv_reference=Vector([-25, 0]), epsilon=0.7,
    #             alpha=alpha, states_to_observe=[(0, 0)], epochs=epochs, integer_mode=True, graph_types=graph_types,
    #             number_of_agents=number_of_agents, agents_configuration=agents_configuration, gamma=gamma,
    #             max_steps=max_steps)
    #
    # agents_configuration = {**agent_a1_configuration, **agent_pql_configuration}
    #
    # test_agents(environment=BonusWorldAcyclic(), hv_reference=Vector([-50, -50, -50]), epsilon=0.25, alpha=alpha,
    #             states_to_observe=[(0, 0)], epochs=epochs, integer_mode=True, graph_types=graph_types,
    #             number_of_agents=number_of_agents, agents_configuration=agents_configuration, gamma=gamma,
    #             max_steps=max_steps)
    #
    # agents_configuration = {**agent_a1_configuration}
    #
    # test_agents(environment=PressurizedBountifulSeaTreasureRightDownStochastic(), hv_reference=Vector([-25, 0, -100]),
    #             epsilon=0.7, alpha=alpha, states_to_observe=[(0, 0)], epochs=epochs, integer_mode=True,
    #             graph_types=graph_types, number_of_agents=number_of_agents, agents_configuration=agents_configuration,
    #             gamma=gamma, max_steps=max_steps)


if __name__ == '__main__':
    main()
