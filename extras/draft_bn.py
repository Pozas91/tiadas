"""
Example of BN agent, dumping data into yaml file, with the environment ResourceGathering, specific precision and
binary dumps
"""
import time

import utils.miscellaneous as um
from agents.agent_bn import AgentBN
from configurations.paths import dumps_path
from environments import Environment, ResourceGatheringEpisodic
from models import GraphType, Vector


def dumps(data: dict, environment: Environment, **kwargs):
    """
    Dumps full_data given into dumps directory
    :param environment:
    :param data:
    :return:
    """

    timestamp = int(time.time())

    # Get environment name in snake case
    environment = um.str_to_snake_case(environment.__class__.__name__)

    # Get only first letter of each word
    env_name_abbr = ''.join([word[0] for word in environment.split('_')])

    columns = kwargs.get('columns')

    if columns:
        # Specify full path
        file_path = dumps_path.joinpath(
            'bn/train_data/{}_bn_{}_{}_{}.yml'.format(env_name_abbr, timestamp, Vector.decimal_precision, columns)
        )
    else:
        # Specify full path
        file_path = dumps_path.joinpath(
            'bn/train_data/{}_bn_{}_{}.yml'.format(env_name_abbr, timestamp, Vector.decimal_precision)
        )

    # If any parents doesn't exist, make it.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open(mode='w+', encoding='UTF-8') as f:
        f.write(um.structures_to_yaml(data=data))


def main():
    # Define gamma
    gamma = .9
    # Each 30 sweeps make it a dump
    sweeps_dumps = 30

    for decimal_precision in [0.01, 0.005]:
        # Set numbers of decimals allowed
        Vector.set_decimal_precision(decimal_precision=decimal_precision)

        # Define same tolerance that decimal precision, but is possible to change
        tolerance = decimal_precision

        # Create environment
        environment = ResourceGatheringEpisodic()

        # Create agent
        agent_bn = AgentBN(environment=environment, convergence_graph=False, gamma=gamma)

        # Time train
        t0 = time.time()

        print('Training with tolerance: {}...'.format(tolerance))

        agent_bn.train(graph_type=GraphType.SWEEP, tolerance=tolerance, sweeps_dump=sweeps_dumps)

        # Calc total time
        total_time = time.time() - t0

        # Convert to vectors
        vectors = {key: [vector.tolist() for vector in vectors] for key, vectors in agent_bn.v.items()}

        # Prepare full_data to dumps
        data = {
            'time': '{}s'.format(total_time),
            'memory': {
                'v_s_0': len(agent_bn.v[environment.initial_state]),
                'full': sum(len(vectors) for vectors in agent_bn.v.values())
            },
            'vectors': vectors
        }

        # Configuration of environment
        environment_info = vars(environment).copy()
        environment_info.pop('_action_space', None)
        environment_info.pop('np_random', None)

        # Configuration of agent
        agent_info = {
            'gamma': agent_bn.gamma,
            'initial_q_value': agent_bn.initial_q_value,
            'initial_seed': agent_bn.initial_seed,
            'interval_to_get_data': agent_bn.interval_to_get_data,
            'total_sweeps': agent_bn.total_sweeps,
            'tolerance': tolerance
        }

        # Extra data
        data.update({'environment': environment_info})
        data.update({'agent': agent_info})

        # Dumps partial execution
        dumps(data=data, environment=environment)

        # Dump binary information
        agent_bn.save()


if __name__ == '__main__':
    main()
