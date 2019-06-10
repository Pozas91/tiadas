from pathlib import Path

import math
import matplotlib.pyplot as plt
import numpy as np

import utils.miscellaneous as um
from agents import AgentA1
from gym_tiadas.gym_tiadas.envs import DeepSeaTreasureRightDownStochastic, \
    PressurizedBountifulSeaTreasureRightDownStochastic
from models import Vector, GraphType


def experiment_dps_with_agent_a1(environment_variant, columns=range(1, 11), epsilon: float = 0.7, alpha: float = 0.8,
                                 states_to_observe=None, hv_reference: Vector = Vector([-25, 0]),
                                 graph_types: tuple = (GraphType.STEPS, GraphType.EPOCHS), number_of_agents: int = 50):
    # Parameters
    if states_to_observe is None:
        states_to_observe = [(0, 0)]

    hv_reference *= Vector.decimals
    evaluation_mechanisms = ('C-PQL', 'PO-PQL', 'HV-PQL')
    env_name = environment_variant().__class__.__name__
    env_name_snake = um.str_to_snake_case(env_name)

    sub_problems_graph = dict()

    path = '.\\dumps\\experiments_a1\\graphs\\{}_{}_{}_{}.{}'
    config_path = '.\\dumps\experiments_a1\\graphs\\{}.{}'

    sub_problems_graph.update({
        'STEPS': {i_columns: list() for i_columns in columns},
        'EPOCHS': {i_columns: list() for i_columns in columns},
    })

    filename_config = Path(
        config_path.format(env_name_snake, 'config').lower()
    )

    with filename_config.open(mode='w+') as file:
        file_data = "seeds = [{}]\n".format(','.join(range(number_of_agents)))
        file_data += "epsilon = {}\n".format(epsilon)
        file_data += "alpha = {}\n".format(alpha)
        file_data += "hv_reference = [{}]\n".format(','.join(map(str, hv_reference.components)))

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
                environment = environment_variant(columns=i_columns, seed=seed)

                # Multiply by Vector.decimals to work with int numbers allowing some decimals
                for state, reward in environment.finals.items():
                    environment.finals.update({state: reward * Vector.decimals})

                # Multiply default reward
                environment.default_reward *= Vector.decimals

                # Build an instance of agent
                agent = AgentA1(environment=environment, epsilon=epsilon, alpha=alpha,
                                states_to_observe=states_to_observe, evaluation_mechanism=evaluation_mechanism,
                                hv_reference=hv_reference, seed=seed, graph_types=graph_types)

                # Pareto frontier
                pareto_frontier = [
                    Vector(pareto_point) * Vector.decimals for pareto_point in environment.pareto_optimal[0:i_columns]
                ]

                # Train the agent
                # agent.objective_training(list_of_vectors=pareto_frontier)
                agent.train(epochs=100)

                # Restore STEPS last data
                last_data = sub_problems_graph.get('STEPS').get(i_columns)
                last_data.append(agent.total_steps)
                sub_problems_graph.get('STEPS').update({i_columns: last_data})

                # Restore STEPS last data
                last_data = sub_problems_graph.get('EPOCHS').get(i_columns)
                last_data.append(agent.total_epochs)
                sub_problems_graph.get('EPOCHS').update({i_columns: last_data})

                # Save trained model
                # filename = agent.json_filename()
                # agent.save(filename='{}\\{}'.format('experiments_a1', filename))

                for graph in graphs:
                    # Recover old data
                    data = graphs.get(graph).get(evaluation_mechanism)

                    # Prepare new data
                    agent_data = agent.states_to_observe.get(graph).get(states_to_observe[0])
                    raw_data = np.divide(agent_data, Vector.decimals)
                    data.append(raw_data)

                    # Update data in the dictionary
                    graphs.get(graph).update({
                        evaluation_mechanism: data
                    })

        for graph in graphs:

            for evaluation_mechanism in evaluation_mechanisms:
                # Recover old data
                data = graphs.get(graph).get(evaluation_mechanism)

                # Calc max length data
                epochs = max([len(x) if len(x) > 0 else 1 for x in data])

                # Change to same length at all arrays
                for i, x_steps in enumerate(data):
                    # Length of x
                    len_x = len(x_steps)

                    # Difference
                    difference = epochs - len_x

                    # If x is not empty
                    if len_x > 0:
                        data[i] = np.append(x_steps, [x_steps[-1]] * difference)
                    else:
                        data[i] = [0] * difference

                process_data = np.average(data, axis=0)
                error = np.std(data, axis=0)
                x = np.arange(0, epochs, 1)

                # Set graph to current evaluation mechanism
                plt.errorbar(x=x, y=process_data, yerr=error, errorevery=math.ceil(epochs * 0.1),
                             label=evaluation_mechanism)

                evaluation_mechanism_name = evaluation_mechanism.split('-')[0]
                graph_name = graph.name

                filename_m = Path(
                    path.format(env_name_snake, i_columns, graph_name, evaluation_mechanism_name, 'm').lower()
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
                plt.xlabel('{} x{}'.format(graph.name, AgentA1.steps_to_calc_hypervolume))
            elif graph == GraphType.TIME:
                plt.xlabel('{} x{}s'.format(graph.name, AgentA1.seconds_to_calc_hypervolume))
            else:
                plt.xlabel(graph.name)

            plt.ylabel('HV max')
            plt.title('{} environment {}'.format(env_name, i_columns))
            plt.legend(loc='upper left')
            plt.show()

    # STEPS GRAPH
    x_steps = list(sub_problems_graph.get('STEPS').keys())
    raw_data_steps = list(sub_problems_graph.get('STEPS').values())
    data_steps = np.average(raw_data_steps, axis=1)
    error_steps = np.std(raw_data_steps, axis=1)

    x_epochs = list(sub_problems_graph.get('EPOCHS').keys())
    raw_data_epochs = list(sub_problems_graph.get('EPOCHS').values())
    data_epochs = np.average(raw_data_epochs, axis=1)
    error_epochs = np.std(raw_data_epochs, axis=1)

    plt.errorbar(x=x_steps, y=data_steps, yerr=error_steps, errorevery=math.ceil(len(x_steps) * 0.1), label='STEPS')
    plt.errorbar(x=x_epochs, y=data_epochs, yerr=error_epochs, errorevery=math.ceil(len(x_epochs) * 0.1),
                 label='EPOCHS')

    plt.xlabel("# Columns taken")
    plt.ylabel('#')
    plt.legend(loc='upper left')
    plt.show()


def main():
    # environment_variant = DeepSeaTreasureRightDown
    environment_variant = DeepSeaTreasureRightDownStochastic
    # environment_variant = PressurizedBountifulSeaTreasureRightDownStochastic

    # hv_reference = Vector([-25, 0, -120])
    hv_reference = Vector([-25, 0])

    experiment_dps_with_agent_a1(environment_variant=environment_variant, columns=range(1, 6), alpha=0.8,
                                 hv_reference=hv_reference, number_of_agents=20)


if __name__ == '__main__':
    main()
