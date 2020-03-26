import time
from pathlib import Path

import utils.miscellaneous as um
from agents import AgentW
from environments import Environment, ResourceGathering, ResourceGatheringEpisodic
from models import GraphType, Vector


def dumps(data: dict, environment: Environment):
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

    # Specify full path
    file_path = Path(__file__).parent.parent.joinpath(
        'dumps/w/train_data/{}_w_{}_{}.yml'.format(env_name_abbr, timestamp, Vector.decimal_precision)
    )

    # If any parents doesn't exist, make it.
    file_path.parent.mkdir(parents=True, exist_ok=True)

    with file_path.open(mode='w+', encoding='UTF-8') as f:
        f.write(um.structures_to_yaml(data=data))


def draft_w():
    tolerance = 0.00001

    for decimal_precision in [0.05, 0.005, 0.001]:
        # Create environment
        # environment = ResourceGatheringEpisodicSimplified()
        # environment = ResourceGatheringSimplified()
        environment = ResourceGatheringEpisodic()

        # Create agent
        agent_w = AgentW(environment=environment, convergence_graph=True, gamma=.9)

        # Time train
        t0 = time.time()

        # Set numbers of decimals allowed
        Vector.set_decimal_precision(decimal_precision=decimal_precision)

        agent_w.train(graph_type=GraphType.SWEEP, limit=1)

        # Calc total time
        total_time = time.time() - t0

        # Convert to vectors
        vectors = {key: [vector.tolist() for vector in vectors] for key, vectors in agent_w.v.items()}

        # Prepare full_data to dumps
        data = {
            'time': '{}s'.format(total_time),
            'memory': {
                'v_s_0': len(agent_w.v[environment.initial_state]),
                'full': sum(len(vectors) for vectors in agent_w.v.values())
            },
            'vectors': vectors
        }

        # Configuration of environment
        environment_info = vars(environment).copy()
        environment_info.pop('_action_space', None)
        environment_info.pop('np_random', None)

        # Configuration of agent
        agent_info = {
            'gamma': agent_w.gamma,
            'initial_q_value': agent_w.initial_q_value,
            'initial_seed': agent_w.initial_seed,
            'interval_to_get_data': agent_w.interval_to_get_data,
            'max_steps': agent_w.max_iterations,
            'total_sweeps': agent_w.total_sweeps,
            'tolerance': tolerance
        }

        # Extra data
        data.update({'environment': environment_info})
        data.update({'agent': agent_info})

        # Dumps partial execution
        dumps(data=data, environment=environment)


if __name__ == '__main__':
    draft_w()
