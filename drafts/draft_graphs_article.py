import os
from pathlib import Path

import yaml

import utils.hypervolume as uh
from models import Vector, AgentType

# line_config = {
#     'B': {
#         'color': 'r',
#         'marker': 'state'
#     },
#     'W_{0.1}': {
#         'color': 'b',
#         'marker': '.'
#     },
#     'W_{0.05}': {
#         'color': 'b',
#         'marker': 'd'
#     },
#     'W_{0.02}': {
#         'color': 'b',
#         'marker': 'o'
#     },
#     'W_{0.01}': {
#         'color': 'b',
#         'marker': '+'
#     },
#     'W_{0.005}': {
#         'color': 'b',
#         'marker': 'r'
#     },
#     'W_{0.001}': {
#         'color': 'b',
#         'marker': 'x'
#     }
# }

line_config = {
    'B': {
        'color': 'r',
        'marker': '+'
    },
    'MPQ_{1}': {
        'color': 'k',
        'marker': 'o'
    },
    'MPQ_{0.1}': {
        'color': 'm',
        'marker': 'd'
    },
    'MPQ_{0.05}': {
        'color': 'y',
        'marker': 'state'
    },
    'MPQ_{0.02}': {
        'color': 'g',
        'marker': '*'
    },
    'MPQ_{0.01}': {
        'color': 'c',
        'marker': '^'
    },
    'MPQ_{0.005}': {
        'color': 'b',
        'marker': 'x'
    },
    'MPQ_{0.001}': {
        'color': 'k',
        'marker': 'h'
    },
    'W_{1}': {
        'color': 'k',
        'marker': 'h'
    },
    'W_{0.1}': {
        'color': 'm',
        'marker': 'x'
    },
    'W_{0.05}': {
        'color': 'y',
        'marker': '*'
    },
    'W_{0.02}': {
        'color': 'g',
        'marker': '^'
    },
    'W_{0.01}': {
        'color': 'c',
        'marker': 'state'
    },
    'W_{0.005}': {
        'color': 'b',
        'marker': 'd'
    },
    'W_{0.001}': {
        'color': 'k',
        'marker': 'o'
    }
}

vector_reference = Vector((-25, 0))


def pareto_graph(data: dict):
    # Columns
    columns = list(data.keys())[0]

    # Prepare hypervolume to dumps data
    pareto_file = Path(__file__).parent.joinpath('article/output/pareto_{}.m'.format(columns))

    # If any parents doesn't exist, make it.
    pareto_file.parent.mkdir(parents=True, exist_ok=True)

    data = data[columns]

    with pareto_file.open(mode='w+', encoding='UTF-8') as file:
        file_data = 'figure;\n'
        file_data += 'hold on;\n\n'

        file_data += "title('Pareto Frontier ({})');\n\n".format(columns)

        labels = dict()

        for label, information in data.items():
            # Calculate hypervolume
            element = information['vectors']['(0, 0)']

            # Update labels information
            labels.update({label: element})

        for label, information in labels.items():
            file_data += 'X = [{}];\n'.format(', '.join(str(info[0]) for info in information))
            file_data += 'Y = [{}];\n'.format(', '.join(str(info[1]) for info in information))
            file_data += "scatter(X, Y, [], '{}', '{}');\n\n".format(
                line_config[label]['color'], line_config[label]['marker']
            )

        file_data += 'legend({});\n'.format(', '.join("'{}'".format(label) for label in labels.keys()))
        file_data += 'hold off;\n'

        file.write(file_data)


def hv_graph(data: dict):
    # Prepare hypervolume to dumps data
    hv_file = Path(__file__).parent.joinpath('article/output/hv.m')

    # If any parents doesn't exist, make it.
    hv_file.parent.mkdir(parents=True, exist_ok=True)

    with hv_file.open(mode='w+', encoding='UTF-8') as file:
        file_data = 'figure;\n'
        file_data += 'hold on;\n\n'

        file_data += "title('Hypervolume');\n\n"

        labels = dict()

        for columns, values in data.items():

            for label, information in values.items():
                # Convert to vectors
                vectors = list(map(Vector, information['vectors']['(0, 0)']))

                # Calculate hypervolume
                element = uh.calc_hypervolume(vectors=vectors, reference=vector_reference)

                # Get previous data
                previous_data = labels.get(label, {columns: element})

                if columns not in previous_data:
                    previous_data.update({columns: element})

                # Update labels information
                labels.update({label: previous_data})

        for label, information in labels.items():
            file_data += 'X = [{}];\n'.format(', '.join(map(str, information.keys())))
            file_data += 'Y = [{}];\n'.format(', '.join(map(str, information.values())))
            file_data += "plot(X, Y, 'Color', '{}', 'Marker', '{}');\n\n".format(
                line_config[label]['color'], line_config[label]['marker']
            )

        file_data += "x_label('# of diagonals');\n"
        file_data += "y_label('Hypervolume');\n"
        file_data += "\n"
        file_data += 'legend({});\n'.format(', '.join("'{}'".format(label) for label in labels.keys()))
        file_data += 'hold off;\n'

        file.write(file_data)


def time_graph(data: dict):
    # Prepare time to dumps data
    time_file = Path(__file__).parent.joinpath('article/output/time.m')

    # If any parents doesn't exist, make it.
    time_file.parent.mkdir(parents=True, exist_ok=True)

    with time_file.open(mode='w+', encoding='UTF-8') as file:
        file_data = 'figure;\n'
        file_data += 'hold on;\n\n'

        file_data += "title('Time');\n\n"

        labels = dict()

        for columns, values in data.items():

            for label, information in values.items():
                # Extract memory information
                element = information['time']

                # Get previous data
                previous_data = labels.get(label, {columns: element})

                if columns not in previous_data:
                    previous_data.update({columns: element})

                # Update labels information
                labels.update({label: previous_data})

        for label, information in labels.items():
            file_data += 'X = [{}];\n'.format(', '.join(map(str, information.keys())))
            file_data += 'Y = [{}];\n'.format(', '.join(map(str, information.values())))
            file_data += "plot(X, Y, 'Color', '{}', 'Marker', '{}');\n\n".format(
                line_config[label]['color'], line_config[label]['marker']
            )

        file_data += "x_label('# of diagonals');\n"
        file_data += "y_label('Seconds');\n"
        file_data += "\n"
        file_data += 'legend({});\n'.format(', '.join("'{}'".format(label) for label in labels.keys()))
        file_data += 'hold off;\n'

        file.write(file_data)


def memory_graph(data: dict):
    # Prepare memory to dumps data
    memory_file = Path(__file__).parent.joinpath('article/output/memory.m')

    # If any parents doesn't exist, make it.
    memory_file.parent.mkdir(parents=True, exist_ok=True)

    with memory_file.open(mode='w+', encoding='UTF-8') as file:
        file_data = 'figure;\n'
        file_data += 'hold on;\n\n'

        file_data += "title('Memory');\n\n"

        labels = dict()

        for columns, values in data.items():

            for label, information in values.items():

                # Extract memory information
                element = information['memory']['full']

                # Get previous data
                previous_data = labels.get(label, {columns: element})

                if columns not in previous_data:
                    previous_data.update({columns: element})

                # Update labels information
                labels.update({label: previous_data})

        for label, information in labels.items():
            file_data += 'X = [{}];\n'.format(', '.join(map(str, information.keys())))
            file_data += 'Y = [{}];\n'.format(', '.join(map(str, information.values())))
            file_data += "plot(X, Y, 'Color', '{}', 'Marker', '{}');\n\n".format(
                line_config[label]['color'], line_config[label]['marker']
            )

        file_data += "x_label('# of diagonals');\n"
        file_data += "y_label('# of vectors');\n"
        file_data += "\n"
        file_data += 'legend({});\n'.format(', '.join("'{}'".format(label) for label in labels.keys()))
        file_data += 'hold off;\n'

        file.write(file_data)


def memory_pareto_graph(data: dict):
    # Prepare memory to dumps data
    pareto_memory_file = Path(__file__).parent.joinpath('article/output/pareto_memory.m')

    # If any parents doesn't exist, make it.
    pareto_memory_file.parent.mkdir(parents=True, exist_ok=True)

    with pareto_memory_file.open(mode='w+', encoding='UTF-8') as file:
        file_data = 'figure;\n'
        file_data += 'hold on;\n\n'

        file_data += "title('Memory V_{s0}');\n\n"

        labels = dict()

        for columns, values in data.items():

            for label, information in values.items():

                # Extract memory information
                element = information['memory']['v_s_0']

                # Get previous data
                previous_data = labels.get(label, {columns: element})

                if columns not in previous_data:
                    previous_data.update({columns: element})

                # Update labels information
                labels.update({label: previous_data})

        for label, information in labels.items():
            file_data += 'X = [{}];\n'.format(', '.join(map(str, information.keys())))
            file_data += 'Y = [{}];\n'.format(', '.join(map(str, information.values())))
            file_data += "plot(X, Y, 'Color', '{}', 'Marker', '{}');\n\n".format(
                line_config[label]['color'], line_config[label]['marker']
            )

        file_data += "x_label('# of diagonals');\n"
        file_data += "y_label('# of vectors');\n"
        file_data += "\n"
        file_data += 'legend({});\n'.format(', '.join("'{}'".format(label) for label in labels.keys()))
        file_data += 'hold off;\n'

        file.write(file_data)


def main():
    # Read from
    article_directory = Path(__file__).parent.joinpath('article')

    # Extract all files from input directory
    files = filter(Path.is_file, os.scandir(article_directory))

    full_data = dict()

    # Prepare data
    for file in files:

        # Extract parts of path name
        file_name = file.name.split('.yml')[0].split('_')

        # Extract type of agent
        type_of_agent: AgentType = AgentType.from_string(file_name[1])

        # Extract number of diagonals from path
        columns = int(file_name[-1])

        # Read full_data from path
        yaml_content = yaml.load(Path(file.path).open(mode='r'), Loader=yaml.FullLoader)

        # Reformat time
        yaml_content['time'] = float(yaml_content['time'])

        # Delete diagonals attribute from yaml
        yaml_content.pop('columns', None)

        if type_of_agent is AgentType.B:
            label = 'B'
        elif type_of_agent is AgentType.W:
            decimals = file_name[-2]
            label = "W_{{{}}}".format(decimals)
        elif type_of_agent is AgentType.MPQ:
            decimals = file_name[-2]
            label = "MPQ_{{{}}}".format(decimals)
        else:
            raise ValueError("Type of agents doesn't recognize")

        previous_data = full_data.get(columns, {})
        previous_data.update({
            label: yaml_content
        })

        full_data.update({columns: previous_data})

    # Order dictionary
    full_data = {key: full_data[key] for key in sorted(full_data.keys())}

    memory_graph(data=full_data)
    memory_pareto_graph(data=full_data)
    time_graph(data=full_data)
    hv_graph(data=full_data)
    # pareto_graph(data=full_data)


if __name__ == '__main__':
    main()
