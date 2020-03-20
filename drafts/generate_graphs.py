import os
from pathlib import Path

# Read from
import yaml


def main():
    # Initial state
    initial_state = '(0, 0)'

    input_directory = Path(__file__).parent.joinpath('input')

    # If any parents doesn't exist, make it.
    input_directory.mkdir(parents=True, exist_ok=True)

    # Extract all files from input directory
    files = filter(Path.is_file, os.scandir(input_directory))

    graph_vectors = dict()
    graph_time = dict()
    graph_memory = dict()

    for file in files:

        # Extract parts of path name
        file_name = file.name.split('.yml')[0].split('_')

        cols = file_name[-1]

        key = 'Columns: {} - Agent: {} - Decimals: {}'.format(cols, file_name[1], file_name[-2])

        graph_vectors.update({
            key: {
                'X': [],
                'Y': []
            }
        })

        graph_time.update({
            key: {
                'X': [],
                'Y': []
            }
        })

        graph_memory.update({
            key: {
                'X': [],
                'Y': []
            }
        })

        with Path(file).open(mode='r', encoding='UTF-8') as f:

            # Load all data from file
            data = yaml.load(f, Loader=yaml.FullLoader)

            # Information for graph time
            graph_time[key]['X'].append(cols)
            graph_time[key]['Y'].append(data['time'])

            # Information for graph memory
            graph_memory[key]['X'].append(cols)
            graph_memory[key]['Y'].append(data['memory']['v_s_0'])

            # Information for graph vectors
            for vectors in data['vectors'][initial_state]:
                graph_vectors[key]['X'].append(vectors[0])
                graph_vectors[key]['Y'].append(vectors[1])

        title = 'DSTRD - Stochastic {}'

        # Generate vectors graph
        generate_vectors(
            graph_data=graph_vectors, output_name='vectors', title=title, x_label='Steps', y_label='Treasure'
        )

        # Generate time graph
        generate_vectors(
            graph_data=graph_time, output_name='time', title=title, x_label='Cols', y_label='Time in seconds'
        )

        # Generate memory graph
        generate_vectors(
            graph_data=graph_memory, output_name='memory', title=title, x_label='Cols', y_label='# Vectors'
        )


def generate_vectors(graph_data: dict, output_name: str, title: str, x_label: str, y_label: str):
    # Define title
    title = title.format(output_name.upper())
    # Define output file
    output_file = Path(__file__).parent.joinpath('output/{}.m'.format(output_name))
    # If any parents doesn't exist, make it.
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with output_file.open(mode='w+', encoding='UTF-8') as f:
        file_data = 'figure;\n'
        file_data += 'hold on;\n\n'

        labels = list()

        for label, values in graph_data.items():
            file_data += 'X = [{}];\n'.format(', '.join(map(str, values['X'])))
            file_data += 'Y = [{}];\n'.format(', '.join(map(str, values['Y'])))
            # file_data += 'scatter(X, Y, [], \'red\', \'state\');\n\n'
            file_data += 'scatter(X, Y);\n\n'
            labels.append(label)

        file_data += 'xlabel(\'{}\');\n'.format(x_label)
        file_data += 'ylabel(\'{}\');\n'.format(y_label)
        file_data += 'title(\'{}\');\n'.format(title)
        file_data += 'legend({});\n'.format(', '.join("'{}'".format(label) for label in labels))
        file_data += 'hold off;\n'

        f.write(file_data)


if __name__ == '__main__':
    main()
