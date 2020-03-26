import time

import yaml

import utils.miscellaneous as um
import utils.models as u_models
from agents import AgentMPQ
from configurations.paths import dumps_path
from environments import Environment, DeepSeaTreasureRightDownStochastic
from models import GraphType, Vector


def dumps(data: dict, columns: int, environment: Environment):
    """
    Dumps full_data given into dumps directory
    :param environment:
    :param columns:
    :param data:
    :return:
    """

    timestamp = int(time.time())

    # Get environment name in snake case
    environment = um.str_to_snake_case(environment.__class__.__name__)

    # Get only first letter of each word
    env_name_abbr = ''.join([word[0] for word in environment.split('_')])

    # Specify full path
    file_path = dumps_path.joinpath(
        'mpq/train_data/{}_{}_{}_{}.yml'.format(env_name_abbr, timestamp, Vector.decimal_precision, columns)
    )

    # If any parents doesn't exist, make it.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open(mode='w+', encoding='UTF-8') as f:
        f.write(um.structures_to_yaml(data=data))


def train_from_zero():
    # Define variables
    limit = int(3e6)
    epsilon = 0.4
    max_steps = 1000
    alpha = 0.1
    gamma = 1
    graph_type = GraphType.EPISODES
    columns_list = range(1, 6)
    decimals = [0.01, 0.05]

    for decimal_precision in decimals:

        # Set vector decimal precision
        Vector.set_decimal_precision(decimal_precision=decimal_precision)

        for columns in columns_list:
            # Environment
            environment = DeepSeaTreasureRightDownStochastic(columns=columns)

            # Create agent
            agent = AgentMPQ(environment=environment, hv_reference=environment.hv_reference, epsilon=epsilon,
                             alpha=alpha, gamma=gamma, max_steps=max_steps)

            # Time train
            t0 = time.time()

            # Show numbers of columns
            print('# of columns: {}'.format(columns))

            # Agent training
            agent.train(graph_type=graph_type, limit=limit)

            # Calc total time
            total_time = time.time() - t0

            prepare_for_dumps(agent, columns, decimal_precision, graph_type, limit, total_time)


def prepare_for_dumps(agent, columns, decimal_precision, graph_type, limit, total_time):
    # Convert to vectors
    vectors = {key: [v.tolist() for v in vectors.values()] for key, vectors in agent.v.items()}

    # Prepare full_data to dumps
    data = {
        'time': total_time,
        'memory': {
            'v_s_0': len(agent.v[agent.environment.initial_state]),
            'full': sum(len(vectors) for vectors in agent.v.values())
        },
        'vectors': vectors
    }

    # Configuration of environment
    environment_info = vars(agent.environment).copy()
    environment_info.pop('_action_space', None)
    environment_info.pop('np_random', None)
    environment_info.update({'columns': columns})

    # Configuration of agent
    agent_info = {
        'alpha': agent.alpha,
        'gamma': agent.gamma,
        'epsilon': agent.epsilon,
        'evaluation_mechanism': str(agent.evaluation_mechanism),
        'initial_q_value': agent.initial_q_value,
        'initial_seed': agent.initial_seed,
        'interval_to_get_data': agent.interval_to_get_data,
        'max_steps': agent.max_steps,
        'total_steps': agent.total_steps,
        'total_episodes': agent.total_episodes,
        'decimal_precision': decimal_precision,
        'hv_reference': agent.hv_reference,
    }

    training_info = {
        'graph_type': str(graph_type),
        'limit': limit
    }

    # Extra data
    data.update({'environment': environment_info})
    data.update({'agent': agent_info})
    data.update({'training': training_info})

    # Dumps partial execution
    dumps(data=data, columns=columns, environment=agent.environment)

    # Dumps model
    agent.save()


def train_from_file():
    # Models Path
    models_path = 'mpq/models/dstrds_1579869395_1.0_4.bin'

    agent: AgentMPQ = u_models.binary_load(path=dumps_path.joinpath(
        models_path
    ))

    # Data Path
    data_path = dumps_path.joinpath('mpq/train_data/dstrds_1579869395_1.0_4.yml')
    data_file = data_path.open(mode='r', encoding='UTF-8')

    # Load yaml from file
    data = yaml.load(data_file, Loader=yaml.FullLoader)

    # Extract relevant data for training
    before_training_execution = float(data['time'])
    decimal_precision = float(data['agent']['decimal_precision'])
    graph_type = GraphType.from_string(data['training']['graph_type'])
    limit = int(data['training']['limit'])
    columns = int(data['environment']['columns'])

    # Set decimal precision
    Vector.set_decimal_precision(decimal_precision=decimal_precision)

    # Time train
    t0 = time.time()

    # Agent training
    agent.train(graph_type=graph_type, limit=limit)

    # Calc total time
    total_time = (time.time() - t0) + before_training_execution

    prepare_for_dumps(agent, columns, decimal_precision, graph_type, limit, total_time)


if __name__ == '__main__':
    # train_from_zero()
    train_from_file()
