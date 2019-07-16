import time
from enum import Enum
from pathlib import Path

import math
import matplotlib.pyplot as plt
import numpy as np

import utils.hypervolume as uh
import utils.miscellaneous as um
from agents import Agent, AgentPQL, AgentMOSP, AgentA1
from gym_tiadas.gym_tiadas.envs import Environment, DeepSeaTreasure, SpaceExplorationAcyclic
from models import Vector, EvaluationMechanism, GraphType


class AgentType(Enum):
    A1 = 'a1'
    PQL = 'pql'
    SCALARIZED = 'scalarized'


def experiment_dst_agent_a1(dst_variant, columns=range(1, 11), epsilon: float = 0.7, alpha: float = 0.8,
                            states_to_observe: list = None, hv_reference: Vector = Vector([-25, 0]),
                            graph_types: set = None, number_of_agents: int = 30, integer_mode: bool = True,
                            epochs: int = 100, max_steps: int = None, evaluation_mechanisms: set = None):
    # Parameters
    if graph_types is None:
        graph_types = {GraphType.STEPS, GraphType.EPOCHS}

    if states_to_observe is None:
        states_to_observe = [(0, 0)]

    if evaluation_mechanisms is None:
        evaluation_mechanisms = {EvaluationMechanism.HV}

    hv_reference = hv_reference.to_decimals()
    env_name = dst_variant().__class__.__name__
    env_name_snake = um.str_to_snake_case(env_name)
    sub_problems_graph = dict()

    # File timestamp
    timestamp = time.time()

    graph_path = '.\\dumps\\a1\\graphs\\{}_{}_{}_{}_{}_{}_{}.m'
    v_s_0_path = '.\\dumps\\a1\\data\\{}_{}_{}.yml'
    config_path = '.\\dumps\experiments_a1\\graphs\\{}_{}_{}_{}.config'

    sub_problems_graph.update({
        GraphType.STEPS:  {i_columns: list() for i_columns in columns},
        GraphType.EPOCHS: {i_columns: list() for i_columns in columns},
    })

    filename_config = Path(
        config_path.format(timestamp, number_of_agents, epochs, env_name_snake).lower()
    )

    with filename_config.open(mode='w+') as file:
        file_data = "seeds = [{}]\n".format(','.join(map(str, range(number_of_agents))))
        file_data += "epsilon = {}\n".format(epsilon)
        file_data += "alpha = {}\n".format(alpha)
        file_data += "hv_reference = [{}]\n".format(','.join(map(str, hv_reference.components)))
        file_data += "epochs = {}\n".format(epochs)

        file.write(file_data)

    for i_columns in columns:

        graphs = {
            graph_type: {
                evaluation_mechanism: list() for evaluation_mechanism in evaluation_mechanisms
            } for graph_type in graph_types
        }

        for evaluation_mechanism in evaluation_mechanisms:

            for i_agent in range(number_of_agents):

                # Set a seed
                seed = i_agent

                # Build an instance of environment
                environment = dst_variant(columns=i_columns, seed=seed)

                # Build an instance of agent
                agent = AgentA1(environment=environment, epsilon=epsilon, alpha=alpha,
                                states_to_observe=states_to_observe, evaluation_mechanism=evaluation_mechanism,
                                hv_reference=hv_reference, seed=seed, graph_types=graph_types, max_steps=max_steps,
                                integer_mode=integer_mode)

                # Pareto frontier
                pareto_frontier = [
                    Vector(pareto_point).to_decimals() if agent.integer_mode else Vector(pareto_point) for pareto_point
                    in environment.pareto_optimal[0:i_columns]
                ]

                # Train the agent
                agent.objective_training(list_of_vectors=pareto_frontier)

                # Non-dominated vectors found in V(s0)
                v_s_0 = list(agent.v.get(agent.environment.initial_state).values())

                v_s_0_filename = Path(
                    v_s_0_path.format(timestamp, i_columns, str(evaluation_mechanism.name), i_agent).lower()
                )

                with v_s_0_filename.open(mode='w+') as file:
                    file_data = 'v_s_0_non_dominated:\n'
                    file_data += '  {}'.format('\n  '.join(map(str, v_s_0)))

                    file.write(file_data)

                # Restore STEPS last data
                last_data = sub_problems_graph.get(GraphType.STEPS).get(i_columns)
                last_data.append(agent.total_steps)
                sub_problems_graph.get(GraphType.STEPS).update({i_columns: last_data})

                # Restore STEPS last data
                last_data = sub_problems_graph.get(GraphType.EPOCHS).get(i_columns)
                last_data.append(agent.total_epochs)
                sub_problems_graph.get(GraphType.EPOCHS).update({i_columns: last_data})

                # Save trained model
                # filename = agent.json_filename()
                # agent.save(filename='{}\\{}'.format('a1', filename))

                for graph in graphs:
                    # Recover old data
                    data = graphs.get(graph).get(evaluation_mechanism)

                    # Prepare new data
                    agent_data = agent.states_to_observe.get(graph).get(states_to_observe[0])
                    data.append(agent_data)

                    # Update data in the dictionary
                    graphs.get(graph).update({
                        evaluation_mechanism: data
                    })

        for graph in graphs:

            for evaluation_mechanism in evaluation_mechanisms:
                # Recover old data
                data = graphs.get(graph).get(evaluation_mechanism)

                # Calc max length data
                data_max_len = max([len(x) if len(x) > 0 else 1 for x in data])

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
                plt.errorbar(x=x, y=process_data, yerr=error, errorevery=math.ceil(data_max_len * 0.1),
                             label=str(evaluation_mechanism.value))

                evaluation_mechanism_name = str(evaluation_mechanism.value).split('-')[0]
                graph_name = graph.name

                filename_m = Path(
                    graph_path.format(timestamp, number_of_agents, epochs, env_name_snake, i_columns, graph_name,
                                      evaluation_mechanism_name).lower()
                )

                with filename_m.open(mode='w+') as file:
                    file_data = "x = [{}]\n".format(', '.join(map(str, x)))
                    file_data += "Y = [\n{}\n]\n".format(';\n'.join([', '.join(map(str, x)) for x in data]))
                    file_data += "means = mean(Y);\n"
                    file_data += 'figure;\n'
                    file_data += 'plot(x, means);\n'
                    file.write(file_data)

            # Show data
            if graph == GraphType.STEPS:
                plt.xlabel('{} x{}'.format(graph.name, AgentA1.steps_to_get_graph_data))
            elif graph == GraphType.TIME:
                plt.xlabel('{} x{}s'.format(graph.name, AgentA1.seconds_to_get_graph_data))
            else:
                plt.xlabel(graph.name)

            plt.ylabel('HV max')
            plt.title('{} environment {}'.format(env_name, i_columns))
            plt.legend(loc='lower right')
            plt.show()

    # STEPS GRAPH
    x_steps = list(sub_problems_graph.get(GraphType.STEPS).keys())
    raw_data_steps = list(sub_problems_graph.get(GraphType.STEPS).values())
    data_steps = np.average(raw_data_steps, axis=1)
    error_steps = np.std(raw_data_steps, axis=1)

    x_epochs = list(sub_problems_graph.get(GraphType.EPOCHS).keys())
    raw_data_epochs = list(sub_problems_graph.get(GraphType.EPOCHS).values())
    data_epochs = np.average(raw_data_epochs, axis=1)
    error_epochs = np.std(raw_data_epochs, axis=1)

    plt.errorbar(x=x_steps, y=data_steps, yerr=error_steps, errorevery=math.ceil(len(x_steps) * 0.1),
                 label=str(GraphType.EPOCHS.value))
    plt.errorbar(x=x_epochs, y=data_epochs, yerr=error_epochs, errorevery=math.ceil(len(x_epochs) * 0.1),
                 label=str(GraphType.EPOCHS.value))

    plt.xlabel("# Columns taken")
    plt.ylabel('#')
    plt.legend(loc='lower right')
    plt.show()


def dst(epsilon: float = 0.7, alpha: float = 0.9, states_to_observe: list = None,
        graph_types: set = None, number_of_agents: int = 30, integer_mode: bool = True,
        evaluation_mechanisms: set = None, max_steps: int = None, hv_reference: Vector = Vector([-25, 0])):
    """"
    Experiments to objective training
    """

    # Parameters
    if graph_types is None:
        graph_types = {GraphType.STEPS, GraphType.EPOCHS}

    if evaluation_mechanisms is None:
        evaluation_mechanisms = {EvaluationMechanism.HV}

    if states_to_observe is None:
        states_to_observe = [(0, 0)]

    # Build environment
    env = DeepSeaTreasure()
    env_name = env.__class__.__name__
    env_name_snake = um.str_to_snake_case(env_name)

    # Pareto's points
    pareto_frontier = DeepSeaTreasure.pareto_optimal

    # File timestamp
    timestamp = time.time()

    graph_path = '.\\dumps\\pql\\graphs\\{}_{}_{}_{}_{}.m'
    config_path = '.\\dumps\\pql\\config\\{}_{}_{}.config'
    v_s_0_path = '.\\dumps\\pql\\data\\{}_{}_{}_{}.yml'

    filename_config = Path(
        config_path.format(timestamp, number_of_agents, env_name_snake).lower()
    )

    with filename_config.open(mode='w+') as file:
        file_data = "seeds = [{}]\n".format(','.join(map(str, range(number_of_agents))))
        file_data += "epsilon = {}\n".format(epsilon)
        file_data += "alpha_to_scalarized = {}\n".format(alpha)

        file.write(file_data)

    # Create graphs structure
    graphs = dict()

    for graph_type in graph_types:

        graph_evaluations = dict()

        for evaluation_mechanism in evaluation_mechanisms:

            if evaluation_mechanism == EvaluationMechanism.PARETO:
                data = [[uh.calc_hypervolume(pareto_frontier, reference=hv_reference)]] * number_of_agents
            else:
                data = list()

            graph_evaluations.update({evaluation_mechanism: data})

        graphs.update({graph_type: graph_evaluations})

    # Data max length
    data_max_len = float('-inf')

    for i_agent in range(number_of_agents):

        print("Iteration: {}".format(i_agent + 1))

        # Set a seed
        seed = i_agent

        for evaluation_mechanism in evaluation_mechanisms - {EvaluationMechanism.PARETO}:

            print("\tEvaluation Mechanism: {}".format(evaluation_mechanism.name))

            # Build an instance of environment
            environment = DeepSeaTreasure(seed=seed)

            if evaluation_mechanism == EvaluationMechanism.SCALARIZED:

                # Set weights
                weights = (.99, .01)

                # Build agent
                agent = AgentMOSP(seed=seed, environment=environment, weights=weights,
                                  states_to_observe=states_to_observe, epsilon=epsilon, alpha=alpha,
                                  graph_types=graph_types, hv_reference=hv_reference, max_steps=max_steps)

                # Search one extreme objective
                objective = agent.process_reward(pareto_frontier[0])
                agent.objective_training(objective=objective)

                # Get p point from agent test
                p = agent.get_accumulated_reward()

                # Add point found to pareto's frontier found
                agent.pareto_frontier_found.append(p)

                # Reset agent to train again with others weights
                agent.reset()

                # Set weights to find another extreme point
                agent.weights = (.01, .99)

                # Search the other extreme objective
                objective = agent.process_reward(pareto_frontier[-1])
                agent.objective_training(objective=objective)

                # Get q point from agent test.
                q = agent.get_accumulated_reward()

                # Add point found to pareto's frontier found
                agent.pareto_frontier_found.append(q)

                # Search pareto points
                agent.calc_frontier_scalarized(p=p, q=q, solutions_known=pareto_frontier)

                # Non-dominated vectors found in V(s0)
                v_s_0 = agent.pareto_frontier_found

            else:

                # Build an instance of agent
                agent = AgentPQL(environment=environment, epsilon=epsilon, seed=seed, hv_reference=hv_reference,
                                 evaluation_mechanism=evaluation_mechanism, graph_types=graph_types,
                                 states_to_observe=states_to_observe, max_steps=max_steps, integer_mode=integer_mode)

                # Train the agent
                agent.objective_training(list_of_vectors=pareto_frontier)

                # Non-dominated vectors found in V(s0)
                v_s_0 = agent.non_dominated_vectors_from_state(state=agent.environment.initial_state)

            # Order vectors by origin Vec(0) nearest
            v_s_0 = um.order_vectors_by_origin_nearest(vectors=v_s_0)

            v_s_0_filename = Path(
                v_s_0_path.format(timestamp, str(evaluation_mechanism.name), i_agent, env_name_snake).lower()
            )

            with v_s_0_filename.open(mode='w+') as file:
                file_data = 'v_s_0_non_dominated:\n'
                file_data += '  {}'.format('\n  '.join(map(str, v_s_0)))

                file.write(file_data)

            for graph in graphs:
                # Recover old data
                data = graphs.get(graph).get(evaluation_mechanism)

                # Prepare new data
                agent_data = agent.states_to_observe.get(graph).get(states_to_observe[0])
                data.append(agent_data)

                # Update data in the dictionary
                graphs.get(graph).update({
                    evaluation_mechanism: data
                })

                data_max_len = max(data_max_len, len(agent_data))

    for graph in graphs:

        for evaluation_mechanism in graphs.get(graph).keys():

            # Recover old data
            data = graphs.get(graph).get(evaluation_mechanism)

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
            plt.errorbar(x=x, y=process_data, yerr=error, errorevery=math.ceil(data_max_len * 0.1),
                         label=evaluation_mechanism.value)

            evaluation_mechanism_name = str(evaluation_mechanism.value).split('-')[0]
            graph_name = graph.name

            filename_m = Path(
                graph_path.format(number_of_agents, env_name_snake, graph_name, evaluation_mechanism_name,
                                  'm').lower()
            )

            with filename_m.open(mode='w+') as file:
                file_data = "x = [{}]\n".format(', '.join(map(str, x)))
                file_data += "Y = [\n{}\n]\n".format(';\n'.join([', '.join(map(str, x)) for x in data]))
                file_data += "means = mean(Y);\n"
                file_data += 'figure;\n'
                file_data += 'plot(x, means);\n'
                file.write(file_data)

        # Show data
        if graph == GraphType.STEPS:
            plt.xlabel('{} x{}'.format(graph.name, AgentPQL.steps_to_get_graph_data))
        elif graph == GraphType.TIME:
            plt.xlabel('{} x{}s'.format(graph.name, AgentPQL.seconds_to_get_graph_data))
        else:
            plt.xlabel(graph.name)

        plt.ylabel('HV max')
        plt.title('{} environment'.format(env_name))
        plt.legend(loc='lower right')
        plt.show()


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


def write_v_s_0_file(timestamp: int, evaluation_mechanism: EvaluationMechanism, i_agent: int, env_name_snake: str,
                     v_s_0: list, agent_type: AgentType):
    """
    Write V(s0) data.
    :param timestamp:
    :param evaluation_mechanism:
    :param i_agent:
    :param env_name_snake:
    :param v_s_0:
    :param agent_type:
    :return:
    """
    # Path to save file
    v_s_0_path = './dumps/{}/data/{}_{}_{}_{}.yml'

    # Order vectors by origin Vec(0) nearest
    v_s_0 = um.order_vectors_by_origin_nearest(vectors=v_s_0)

    v_s_0_filename = Path(
        v_s_0_path.format(str(agent_type.value), timestamp, str(evaluation_mechanism.value), i_agent,
                          env_name_snake).lower()
    )

    with v_s_0_filename.open(mode='w+') as file:
        file_data = 'v_s_0_non_dominated:\n'
        file_data += '  {}'.format('\n  '.join(map(str, v_s_0)))

        file.write(file_data)


def initialize_graph_data(graph_types: set, agents_configuration: dict) -> dict:
    """
    Initialize graph data dictionary

    :param graph_types:
    :param agents_configuration:
    :return:
    """

    # Create graphs structure
    graphs = dict()

    for graph_type in graph_types:

        data_types = dict()

        for agent_type in agents_configuration:

            data_evaluations = dict()

            for evaluation_mechanism in agents_configuration[agent_type]:
                data_evaluations.update({evaluation_mechanism: list()})

            data_types.update({agent_type: data_evaluations})

        graphs.update({graph_type: data_types})

    return graphs


def test_agents(environment: Environment, hv_reference: Vector, epsilon: float = 0.1, alpha: float = 1.,
                states_to_observe: list = None, epochs: int = 1000, integer_mode: bool = False,
                graph_types: set = None, number_of_agents: int = 30, agents_configuration: dict = None,
                gamma: float = 1., max_steps: int = None):
    """
    This method run an experiment with the parameters and environment given
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
    if graph_types is None:
        graph_types = {GraphType.STEPS, GraphType.EPOCHS}

    if states_to_observe is None:
        states_to_observe = [environment.initial_state]

    if agents_configuration is None:
        agents_configuration = dict()

    # Build environment
    env_name = environment.__class__.__name__
    env_name_snake = um.str_to_snake_case(env_name)

    # File timestamp
    timestamp = int(time.time())

    write_config_file(timestamp=timestamp, number_of_agents=number_of_agents, env_name_snake=env_name_snake,
                      seed=','.join(map(str, range(number_of_agents))), epsilon=epsilon, alpha=alpha)

    # Create graphs structure
    graphs = initialize_graph_data(graph_types=graph_types, agents_configuration=agents_configuration)

    # Data max length
    data_max_len = float('-inf')

    for i_agent in range(number_of_agents):

        print("Execution: {}".format(i_agent + 1))

        # Set a seed
        seed = i_agent

        for agent_type in agents_configuration:

            print('\tAgent: {}'.format(agent_type.value))

            for evaluation_mechanism in agents_configuration[agent_type]:

                print('\t\tEvaluation Mechanism: {}'.format(evaluation_mechanism.value), end=' ')

                # Mark of time
                t0 = time.time()

                # Reset environment
                environment.reset()
                environment.seed(seed=seed)

                # Default values
                v_s_0 = None
                agent = None

                if agent_type == AgentType.SCALARIZED:
                    # Set weights
                    weights = (.99, .01)

                    # Build agent
                    agent = AgentMOSP(seed=seed, environment=environment, weights=weights,
                                      states_to_observe=states_to_observe, epsilon=epsilon, alpha=alpha,
                                      graph_types=graph_types, hv_reference=hv_reference, gamma=gamma,
                                      max_steps=max_steps)

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

                elif agent_type == AgentType.PQL:

                    # Build an instance of agent
                    agent = AgentPQL(environment=environment, epsilon=epsilon, seed=seed, hv_reference=hv_reference,
                                     evaluation_mechanism=evaluation_mechanism, graph_types=graph_types,
                                     states_to_observe=states_to_observe, max_steps=max_steps, gamma=gamma,
                                     integer_mode=integer_mode)

                    # Train the agent
                    agent.train(epochs=epochs)

                    # Non-dominated vectors found in V(s0)
                    v_s_0 = agent.non_dominated_vectors_from_state(state=agent.environment.initial_state)

                elif agent_type == AgentType.A1:
                    # Build an instance of agent
                    agent = AgentA1(environment=environment, epsilon=epsilon, seed=seed, hv_reference=hv_reference,
                                    evaluation_mechanism=evaluation_mechanism, graph_types=graph_types,
                                    states_to_observe=states_to_observe, max_steps=max_steps, gamma=gamma,
                                    integer_mode=integer_mode, alpha=alpha)

                    # Train the agent
                    agent.train(epochs=epochs)

                    # Non-dominated vectors found in V(s0)
                    v_s_0 = agent.v_real().get(agent.environment.initial_state,
                                               {0: agent.environment.default_reward.zero_vector}).values()

                else:
                    ValueError("Agent type does not valid!")

                print('-> {:.2f}s'.format(time.time() - t0))

                # Write vectors found into file
                write_v_s_0_file(timestamp=timestamp, evaluation_mechanism=evaluation_mechanism, i_agent=i_agent,
                                 env_name_snake=env_name_snake, v_s_0=v_s_0, agent_type=agent_type)

                # Calc data maximum length
                data_max_len = update_graph(agent=agent, data_max_len=data_max_len,
                                            evaluation_mechanism=evaluation_mechanism, graphs=graphs,
                                            states_to_observe=states_to_observe, agent_type=agent_type)

    prepare_data_and_show_graph(timestamp=timestamp, data_max_len=data_max_len, env_name=env_name,
                                env_name_snake=env_name_snake, graphs=graphs, number_of_agents=number_of_agents,
                                agents_configuration=agents_configuration)


def update_graph(agent: Agent, data_max_len: int, evaluation_mechanism: EvaluationMechanism, graphs: dict,
                 states_to_observe: list, agent_type: AgentType):
    """
    Update graph to show
    :param agent_type:
    :param agent:
    :param data_max_len:
    :param evaluation_mechanism:
    :param graphs:
    :param states_to_observe:
    :return:
    """

    for graph_type in graphs:
        # Recover old data
        data = graphs[graph_type][agent_type][evaluation_mechanism]

        # Prepare new data
        agent_data = agent.states_to_observe[graph_type][states_to_observe[0]]
        data.append(agent_data)

        # Update data in the dictionary
        graphs[graph_type][agent_type].update({
            evaluation_mechanism: data
        })

        data_max_len = max(data_max_len, len(agent_data))

    return data_max_len


def prepare_data_and_show_graph(timestamp: int, data_max_len: int, env_name: str, env_name_snake: str, graphs: dict,
                                number_of_agents: int, agents_configuration: dict):
    """
    Prepare data to show a graph with the information about results
    :param agents_configuration:
    :param timestamp:
    :param data_max_len:
    :param env_name:
    :param env_name_snake:
    :param graphs:
    :param number_of_agents:
    :return:
    """
    graph_path = './dumps/{}/graphs/{}_{}_{}_{}_{}.m'

    for graph_type in graphs:

        for agent_type in graphs[graph_type]:

            for evaluation_mechanism in graphs[graph_type][agent_type].keys():

                # Recover old data
                data = graphs[graph_type][agent_type][evaluation_mechanism]
                color = agents_configuration[agent_type][evaluation_mechanism]

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
                plt.errorbar(x=x, y=process_data, yerr=error, errorevery=math.ceil(data_max_len * 0.1),
                             label='{} {}'.format(agent_type.value, evaluation_mechanism.value), color=color)

                evaluation_mechanism_name = evaluation_mechanism.value
                graph_name = graph_type.value

                filename_m = Path(
                    graph_path.format(agent_type.value, timestamp, number_of_agents, env_name_snake, graph_name,
                                      evaluation_mechanism_name).lower()
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
                plt.xlabel('{} x{}'.format(graph_type.value, AgentPQL.steps_to_get_graph_data))
            elif graph_type == GraphType.TIME:
                plt.xlabel('{} x{}s'.format(graph_type.value, AgentPQL.seconds_to_get_graph_data))
            else:
                plt.xlabel(graph_type.value)

        plt.ylabel('HV max')
        plt.title('{} environment'.format(env_name))
        plt.legend(loc='lower right')
        plt.show()


def main():
    columns = range(1, 11)
    alpha = 1.
    number_of_agents = 30
    epochs = 2000

    agents_configuration = {
        AgentType.A1:  {
            EvaluationMechanism.HV: 'yellow',
            EvaluationMechanism.C:  'orange',
            EvaluationMechanism.PO: 'blue'
        },
        AgentType.PQL: {
            EvaluationMechanism.HV: 'pink',
            EvaluationMechanism.C:  'red',
            EvaluationMechanism.PO: 'green'
        },
        AgentType.SCALARIZED: {
            EvaluationMechanism.SCALARIZED: 'cyan'
        }
    }

    graph_types = {
        GraphType.EPOCHS
    }

    # dst(epsilon=0.7, alpha=alpha, states_to_observe=[(0, 0)], graph_types={GraphType.STEPS},
    #     number_of_agents=number_of_agents, evaluation_mechanisms=evaluation_mechanisms, integer_mode=False)
    #
    # dst(epsilon=0.7, alpha=alpha, states_to_observe=[(0, 0)], graph_types={GraphType.TIME},
    #     number_of_agents=number_of_agents, evaluation_mechanisms=evaluation_mechanisms, integer_mode=False)

    # test_agents(environment=DeepSeaTreasureRightDown(), hv_reference=Vector([-25, 0]), epsilon=0.7,
    #             states_to_observe=[(0, 0)], epochs=epochs, graph_types=graph_types, number_of_agents=number_of_agents,
    #             gamma=1., integer_mode=True, agents_configuration=agents_configuration, alpha=alpha)

    # test_agents(environment=DeepSeaTreasureRightDown(), hv_reference=Vector([-25, 0]), epsilon=0.7,
    #             states_to_observe=[(0, 0)], epochs=epochs, graph_types=graph_types, number_of_agents=number_of_agents,
    #             gamma=1., integer_mode=True, agents_configuration=agents_configuration, alpha=alpha)

    # test_agents(environment=BonusWorldAcyclic(), hv_reference=Vector([-101, -101, -150]), epsilon=0.4,
    #             states_to_observe=[(0, 0)], epochs=epochs, graph_types=graph_types, number_of_agents=number_of_agents,
    #             gamma=1., integer_mode=True, agents_configuration=agents_configuration, alpha=alpha)

    # test_agents(environment=MoPuddleWorldAcyclic(), hv_reference=Vector([-50, -150]), epsilon=0.15,
    #             states_to_observe=[(2, 8)], epochs=epochs, graph_types=graph_types, number_of_agents=number_of_agents,
    #             gamma=1., integer_mode=True, agents_configuration=agents_configuration, alpha=alpha)

    test_agents(environment=SpaceExplorationAcyclic(), hv_reference=Vector([-150, -150]), epsilon=0.4,
                states_to_observe=[(0, 0)], epochs=epochs, graph_types=graph_types, number_of_agents=number_of_agents,
                gamma=1., integer_mode=True, agents_configuration=agents_configuration, alpha=alpha)

    # test_agents(environment=SpaceExploration(), hv_reference=Vector([-100, -150]), epsilon=0.4,
    #                alpha=1, states_to_observe=[(5, 2)], epochs=1000, graph_types={GraphType.TIME}, number_of_agents=30,
    #                evaluation_mechanisms=evaluation_mechanisms, gamma=1., max_steps=20, integer_mode=False)

    ####################################################################################################################

    # evaluation_mechanisms = evaluation_mechanisms - {EvaluationMechanism.SCALARIZED}

    # test_agents(environment=BonusWorld(), hv_reference=Vector([0, 0, -150]), epsilon=0.2,
    #                alpha=1, states_to_observe=[(5, 2)], epochs=epochs, graph_types={GraphType.STEPS},
    #                number_of_agents=number_of_agents, evaluation_mechanisms=evaluation_mechanisms, gamma=1.,
    #                integer_mode=False)
    #
    # test_agents(environment=BonusWorld(), hv_reference=Vector([0, 0, -150]), epsilon=0.2,
    #                alpha=1, states_to_observe=[(0, 0)], epochs=epochs, graph_types={GraphType.TIME},
    #                number_of_agents=number_of_agents, evaluation_mechanisms=evaluation_mechanisms, gamma=1.,
    #                integer_mode=False)

    ####################################################################################################################


if __name__ == '__main__':
    main()
