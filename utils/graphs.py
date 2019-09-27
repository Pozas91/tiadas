import math
import time
from decimal import Decimal as D
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

import utils.hypervolume as uh
import utils.miscellaneous as um
from agents import Agent, AgentPQL
from environments import Environment
from models import AgentType, GraphType, Vector, EvaluationMechanism


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


def write_v_from_initial_state_file(timestamp: int, seed: int, env_name_snake: str, v_s_0: list,
                                    agent_type: AgentType, variable, configuration):
    """
    Write V(s0) data.
    :param configuration:
    :param variable:
    :param timestamp:
    :param seed:
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
        v_s_0_path.format(str(agent_type.value), timestamp, seed, env_name_snake, variable, configuration).lower()
    )

    with v_s_0_filename.open(mode='w+') as file:
        file_data = 'v_s_0_non_dominated:\n'
        file_data += '  {}'.format('\n  '.join(map(str, v_s_0)))

        file.write(file_data)


def initialize_graph_data(graph_types: set, agents_configuration: dict) -> (dict, dict):
    """
    Initialize graph data dictionary

    :param graph_types:
    :param agents_configuration:
    :return:
    """

    # Create graphs structure
    graphs = dict()
    graphs_info = dict()

    for graph_type in graph_types:
        data_types = dict()
        data_types_info = dict()

        for agent_type in agents_configuration:

            data_evaluations = dict()
            data_evaluations_info = dict()

            for configuration in agents_configuration[agent_type].keys():
                data_evaluations.update({
                    '{}'.format(configuration): list()
                })

                data_evaluations_info.update({
                    '{}'.format(configuration): {
                        'time': list(),
                        'steps': list(),
                        'solutions_found': list(),
                        'had_solution_found': list(),
                    }
                })

            data_types.update({
                agent_type: data_evaluations
            })

            data_types_info.update({
                agent_type: data_evaluations_info
            })

        graphs.update({
            graph_type: data_types
        })

        graphs_info.update({
            graph_type: data_types_info
        })

    return graphs, graphs_info


def compare_solution(vectors: list, solution: list) -> (int, bool):
    """
    Return a tuple, where the first element is the number of equals vector, and the second element is a indicator if the
    two lists are equals.
    :param vectors:
    :param solution:
    :return:
    """

    exponent = D(10) ** -(Vector.decimals_allowed - 1)
    equals_counter = list()

    for vector in vectors:

        are_equals = False

        for vector_solution in solution:

            if len(vector) != len(vector_solution):
                continue

            are_equals = np.all([
                D(x).quantize(exponent) == D(y).quantize(exponent)
                for x, y in zip(vector.components.astype(float), vector_solution.components.astype(float))
            ])

            if are_equals:
                break

        equals_counter.append(are_equals)

    # Count Number of equals
    number_of_equals = sum(equals_counter)

    return number_of_equals, number_of_equals == len(solution)


def update_graphs(graphs: dict, agent: Agent, configuration: str, states_to_observe: list, agent_type: AgentType,
                  graphs_info: dict, solution: list = None):
    """
    Update graph to show
    :param graphs_info:
    :param solution:
    :param configuration:
    :param agent_type:
    :param agent:
    :param graphs:
    :param states_to_observe:
    :return:
    """

    for graph_type in graphs:

        # Recover old data
        data = graphs[graph_type][agent_type][configuration]
        data_info = graphs_info[graph_type][agent_type][configuration]

        if graph_type is GraphType.MEMORY:
            agent_data = agent.graph_info[graph_type]
        elif graph_type is GraphType.VECTORS_PER_CELL:
            agent_data = agent.graph_info[graph_type]
        else:

            # Prepare new data
            agent_data = list()
            agent_time = list()
            agent_steps = list()
            agent_solutions_found = list()
            agent_had_solutions_found = list()

            for new_data in agent.graph_info[graph_type][states_to_observe[0]]:
                # Calc hypervolume
                hv = uh.calc_hypervolume(list_of_vectors=new_data['vectors'], reference=agent.hv_reference)

                # Add hypervolume to agent_data
                agent_data.append(hv)

                # If solution is given, compare it
                if solution is not None:
                    number_of_solutions, solution_found = compare_solution(
                        vectors=new_data['vectors'], solution=solution)

                    # Extract information
                    agent_time.append(new_data['time'] if solution_found else None)
                    agent_steps.append(new_data['steps'] if solution_found else None)
                    agent_solutions_found.append(number_of_solutions)
                    agent_had_solutions_found.append(solution_found)

        # Save new data
        data.append(agent_data)

        # Additional information
        data_info['time'].append(min([float('inf') if x is None else x for x in agent_time]))
        data_info['steps'].append(min([float('inf') if x is None else x for x in agent_steps]))
        data_info['solutions_found'].append(max([x for x in agent_solutions_found]))
        data_info['had_solution_found'].append(np.any(agent_had_solutions_found))

        # Update data in the dictionary
        graphs[graph_type][agent_type].update({
            configuration: data
        })

        graphs_info[graph_type][agent_type].update({
            configuration: data_info
        })


def test_steps(environment: Environment, hv_reference: Vector, variable: str, agents_configuration: dict,
               steps_limit: int, epsilon: float = 0.1, alpha: float = 0.1, max_steps: int = None,
               states_to_observe: list = None, integer_mode: bool = False, number_of_agents: int = 30,
               gamma: float = 1., evaluation_mechanism: EvaluationMechanism = EvaluationMechanism.C,
               steps_to_get_data: int = 1):
    """

    :param environment:
    :param hv_reference:
    :param variable:
    :param agents_configuration:
    :param steps_limit:
    :param epsilon:
    :param alpha:
    :param max_steps:
    :param states_to_observe:
    :param integer_mode:
    :param number_of_agents:
    :param gamma:
    :param evaluation_mechanism:
    :param steps_to_get_data:
    :return:
    """

    # Set steps to get data from agent
    Agent.steps_to_get_graph_data = steps_to_get_data

    # Parameters
    if states_to_observe is None:
        states_to_observe = [environment.initial_state]

    # Build environment
    env_name = environment.__class__.__name__
    env_name_snake = um.str_to_snake_case(env_name)

    # File timestamp
    timestamp = int(time.time())

    # Write all information in configuration file
    write_config_file(timestamp=timestamp, number_of_agents=number_of_agents, env_name_snake=env_name_snake,
                      seed=','.join(map(str, range(number_of_agents))), epsilon=epsilon, alpha=alpha,
                      max_steps=max_steps, variable=variable, gamma=gamma, steps_limit=steps_limit)

    # Define graph types
    graph_types = {
        GraphType.STEPS
    }

    # Create graphs structure
    graphs = initialize_graph_data(graph_types=graph_types, agents_configuration=agents_configuration)

    # Data max length
    data_max_len = float('-inf')

    # Show information
    print('Environment: {}'.format(env_name))

    # Execute a iteration with different seed for each agent indicate
    for seed in range(number_of_agents):

        # Show information
        print("\tExecution: {}".format(seed + 1))

        # For each configuration
        for agent_type in agents_configuration:

            # Show information
            print('\t\tAgent: {}'.format(agent_type.value))

            # Extract configuration for that agent
            for configuration in agents_configuration[agent_type].keys():
                # Show information
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
                    'epsilon': epsilon, 'alpha': alpha, 'gamma': gamma, 'max_steps': max_steps,
                    'evaluation_mechanism': evaluation_mechanism
                }

                # Modify current configuration
                parameters.update({variable: configuration})

                # Configuration of PQL Agent

                # Removing useless parameters
                del parameters['alpha']

                # Build an instance of agent
                agent = AgentPQL(environment=environment, seed=seed, hv_reference=hv_reference, graph_types=graph_types,
                                 states_to_observe=states_to_observe, integer_mode=integer_mode, **parameters)

                # Train the agent
                agent.steps_train(steps=steps_limit)

                # Non-dominated vectors found in V(s0)
                v_s_0 = agent.q_set_from_state(state=agent.environment.initial_state)

                print('-> {:.2f}s'.format(time.time() - t0))

                # Write vectors found into file
                write_v_from_initial_state_file(timestamp=timestamp, seed=seed, env_name_snake=env_name_snake,
                                                v_s_0=v_s_0, variable=variable, agent_type=agent_type,
                                                configuration=configuration)

                # Update graphs
                update_graphs(agent=agent, graphs=graphs, configuration=str(configuration),
                              states_to_observe=states_to_observe, agent_type=agent_type)

    prepare_data_and_show_graph(timestamp=timestamp, env_name=env_name, env_name_snake=env_name_snake, graphs=graphs,
                                number_of_agents=number_of_agents, agents_configuration=agents_configuration,
                                alpha=alpha, epsilon=epsilon, gamma=gamma, stop_condition={'steps': steps_limit},
                                max_steps=max_steps, initial_state=environment.initial_state, integer_mode=integer_mode,
                                variable=variable)


def test_time(environment: Environment, hv_reference: Vector, variable: str, agents_configuration: dict,
              execution_time: int, epsilon: float = 0.1, alpha: float = 0.1, max_steps: int = None,
              states_to_observe: list = None, integer_mode: bool = False, number_of_agents: int = 30, gamma: float = 1.,
              evaluation_mechanism: EvaluationMechanism = EvaluationMechanism.C, seconds_to_get_data: int = 1,
              solution: list = None):
    """

    :param solution:
    :param seconds_to_get_data:
    :param environment:
    :param hv_reference:
    :param variable:
    :param agents_configuration:
    :param execution_time:
    :param epsilon:
    :param alpha:
    :param max_steps:
    :param states_to_observe:
    :param integer_mode:
    :param number_of_agents:
    :param gamma:
    :param evaluation_mechanism:
    :return:
    """

    # Set seconds to get data from agent
    Agent.seconds_to_get_graph_data = seconds_to_get_data

    # Parameters
    if states_to_observe is None:
        states_to_observe = [environment.initial_state]

    # Build environment
    env_name = environment.__class__.__name__
    env_name_snake = um.str_to_snake_case(env_name)

    # File timestamp
    timestamp = int(time.time())

    # Write all information in configuration file
    write_config_file(timestamp=timestamp, number_of_agents=number_of_agents, env_name_snake=env_name_snake,
                      seed=','.join(map(str, range(number_of_agents))), epsilon=epsilon, alpha=alpha,
                      max_steps=max_steps, variable=variable, gamma=gamma, execution_time=execution_time)

    # Define graph types
    graph_types = {
        GraphType.TIME
    }

    # Create graphs structure
    graphs, graphs_info = initialize_graph_data(graph_types=graph_types, agents_configuration=agents_configuration)

    # Show information
    print('Environment: {}'.format(env_name))

    # Execute a iteration with different seed for each agent indicate
    for seed in range(number_of_agents):

        # Show information
        print("\tExecution: {}".format(seed + 1))

        # For each configuration
        for agent_type in agents_configuration:

            # Show information
            print('\t\tAgent: {}'.format(agent_type.value))

            # Extract configuration for that agent
            for configuration in agents_configuration[agent_type].keys():
                # Show information
                print('\t\t\t{}: {}'.format(variable, configuration), end=' ')

                # Mark of time
                t0 = time.time()

                # Reset environment
                environment.reset()
                environment.seed(seed=seed)

                # Variable parameters
                parameters = {
                    'epsilon': epsilon, 'alpha': alpha, 'gamma': gamma, 'max_steps': max_steps,
                    'evaluation_mechanism': evaluation_mechanism
                }

                # Modify current configuration
                parameters.update({variable: configuration})

                # Configuration of PQL Agent

                # Removing useless parameters
                del parameters['alpha']

                # Build an instance of agent
                agent = AgentPQL(environment=environment, seed=seed, hv_reference=hv_reference, graph_types=graph_types,
                                 states_to_observe=states_to_observe, **parameters, integer_mode=integer_mode)

                # # Train the agent
                agent.time_train(execution_time=execution_time)

                # Non-dominated vectors found in V(s0)
                v_s_0 = agent.q_set_from_state(state=agent.environment.initial_state)

                print('-> {:.2f}s'.format(time.time() - t0))

                # Write vectors found into file
                write_v_from_initial_state_file(timestamp=timestamp, seed=seed, env_name_snake=env_name_snake,
                                                v_s_0=v_s_0, variable=variable, agent_type=agent_type,
                                                configuration=configuration)

                # Update graph to show
                update_graphs(agent=agent, graphs=graphs, configuration=str(configuration), agent_type=agent_type,
                              states_to_observe=states_to_observe, graphs_info=graphs_info, solution=solution)

    prepare_data_and_show_graph(timestamp=timestamp, env_name=env_name, env_name_snake=env_name_snake, graphs=graphs,
                                number_of_agents=number_of_agents, agents_configuration=agents_configuration,
                                alpha=alpha, epsilon=epsilon, gamma=gamma, stop_condition={'time': execution_time},
                                max_steps=max_steps, initial_state=environment.initial_state,
                                integer_mode=integer_mode, variable=variable, graphs_info=graphs_info)


def prepare_data_and_show_graph(timestamp: int, env_name: str, env_name_snake: str, graphs: dict,
                                number_of_agents: int, agents_configuration: dict, alpha: float, gamma: float,
                                epsilon: float, stop_condition: dict, max_steps: int, initial_state: tuple,
                                integer_mode: bool, variable: str, graphs_info: dict):
    """
    Prepare data to show a graph with the information about results
    :param graphs_info:
    :param stop_condition:
    :param variable:
    :param integer_mode:
    :param initial_state:
    :param max_steps:
    :param alpha:
    :param gamma:
    :param epsilon:
    :param agents_configuration:
    :param timestamp:
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
    parameters = {}

    # Get number of graphs
    number_of_graphs = len(graphs)

    # Graph instances
    fig, axs = plt.subplots(nrows=number_of_graphs, **parameters)

    # Convert to one element array
    if number_of_graphs == 1:
        axs = [axs]

    # Resolution 1440x1080 (4:3)
    fig.set_size_inches(14.4, 10.8)
    fig.set_dpi(244)

    y_limit = -1
    x_limit = -1

    for axs_i, graph_type in enumerate(graphs):

        graph_name = graph_type.value

        print('{}'.format(graph_name))

        for agent_type in graphs[graph_type]:

            for configuration in agents_configuration[agent_type].keys():

                # Data info
                data_info = graphs_info[graph_type][agent_type][str(configuration)]

                # Information about time
                info_time_avg = np.average(data_info['time'])
                info_time_std = np.std(data_info['time'])
                info_time_max = np.max(data_info['time'])
                info_time_min = np.min(data_info['time'])

                print('\tTime:')
                print('\t\t{}_{} & {} & {} & {} & {} \\'.format(agent_type, configuration, info_time_avg, info_time_std,
                                                                info_time_max, info_time_min))

                # Information about steps
                info_steps_avg = np.average(data_info['steps'])
                info_steps_std = np.std(data_info['steps'])
                info_steps_max = np.max(data_info['steps'])
                info_steps_min = np.min(data_info['steps'])

                print('\tSteps:')
                print(
                    '\t\t{}_{} & {} & {} & {} & {} \\'.format(agent_type, configuration, info_steps_avg, info_steps_std,
                                                              info_steps_max, info_steps_min))

                # Information about solutions_found
                info_solutions_found_avg = np.average(data_info['solutions_found'])
                info_solutions_found_std = np.std(data_info['solutions_found'])
                info_solutions_found_max = np.max(data_info['solutions_found'])
                info_solutions_found_min = np.min(data_info['solutions_found'])

                print('\tSolutions found:')
                print(
                    '\t\t{}_{} & {} & {} & {} & {} \\'.format(agent_type, configuration, info_solutions_found_avg,
                                                              info_solutions_found_std, info_solutions_found_max,
                                                              info_solutions_found_min))

                # Information about had solution found
                print('\tHad solution:')
                print('\t\t{}'.format(sum(data_info['had_solution_found'])))

                # Recover old data
                data = graphs[graph_type][agent_type][str(configuration)]
                color = agents_configuration[agent_type][configuration]

                # Calc max length
                data_max_len = max(len(x) for x in data)

                # Change to same length at all arrays (to do the average)
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

                # Processing data
                process_data = np.average(data, axis=0)
                error = np.std(data, axis=0)
                x = np.arange(0, data_max_len, 1)

                # Limit calc limit in x and y axes
                x_limit = max(x_limit, x.max())
                y_limit = max(y_limit, process_data.max())

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

        axs[axs_i].set_ylim([0, math.ceil(y_limit * 1.1)])
        axs[axs_i].set_xlim([0, math.ceil(x_limit * 1.1)])

    fig.suptitle('{} environment'.format(env_name))
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    # Text information
    props = {
        'boxstyle': 'round',
        'facecolor': 'wheat',
        'alpha': 0.5
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

    stop_condition_key = next(iter(stop_condition))

    text_information = '\n'.join(basic_information + [
        r'stop_condition={} - {}$'.format(stop_condition_key, stop_condition[stop_condition_key]),
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