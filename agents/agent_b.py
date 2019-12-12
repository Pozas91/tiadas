import itertools
import json
import time

import utils.miscellaneous as um
import utils.numbers as un
from configurations.paths import dumps_path
from environments import Environment
from models import Vector


class AgentB:

    def __init__(self, environment: Environment, limited_precision: bool = False, dumps_format: str = 'yml'):
        """
        :param environment:
        """
        self.environment = environment
        self.states_vectors = dict()
        self.initial_time = None
        self.limited_precision = limited_precision

        # Check if specifies a correct format
        # assert dumps_format in ('json', 'yml')
        assert dumps_format in ('yml',)
        self.dumps_format = dumps_format

    def simulate(self):
        # Time train
        self.initial_time = time.time()

        # Get all states ordered from last to begin
        for state in self.environment.sorted_states(reverse=True):
            # Calc average reward and set to the state
            vectors = self.simulate_state(state=state)

            # Update states average reward dictionary
            self.states_vectors.update({state: vectors})

        self.dumps()

    def simulate_state(self, state: object) -> set:
        # Set current state
        self.environment.current_state = state

        # Set of vectors
        vectors = set()

        # For each possible action
        for action in self.environment.action_space:

            # Get all reachable states
            reachable_states = self.environment.reachable_states(state=state, action=action)

            # Set of vectors
            total_vectors = set()

            # Associate states and vectors
            associate_states = list()
            associate_vectors = list()

            for reachable_state in reachable_states:

                # If next_state is unknown create with a zero vector set.
                if reachable_state not in self.states_vectors:
                    self.states_vectors.update({reachable_state: {self.environment.default_reward.zero_vector}})

                # Calculate reward
                reward = self.environment.transition_reward(state=state, action=action, next_state=reachable_state)

                # Get previous vectors
                accumulated_vectors = set(
                    map(
                        lambda x: x + reward,
                        self.states_vectors[reachable_state]
                    )
                )

                associate_states.append(reachable_state)
                associate_vectors.append(accumulated_vectors)

                # Add current vectors to total vectors
                total_vectors = total_vectors.union(accumulated_vectors)

            self.states_vectors.update({state: total_vectors})

            # For each next state
            for product_vectors in itertools.product(*associate_vectors):

                # Extract zero vector
                vector = self.environment.default_reward.zero_vector

                for i, reward in enumerate(product_vectors):
                    # Next state
                    reachable_state = associate_states[i]

                    # Calculate probability
                    probability = self.environment.transition_probability(
                        state=state, action=action, next_state=reachable_state
                    )

                    # Calc total vector
                    vector += (reward * probability)

                # Add to set of vectors
                vectors.add(vector)

        if self.limited_precision:
            vectors = map(lambda x: un.round_with_precision(x, Vector.decimal_precision), vectors)

        # Return all vectors found
        return set(Vector.m3_max(list(vectors)))

    def reset(self):
        self.environment.reset()
        self.states_vectors = dict()
        self.initial_time = None

    def dumps(self):
        """
        Dumps agent data into dumps directory
        :return:
        """

        # Calc total time
        total_time = time.time() - self.initial_time

        if self.dumps_format == 'yml':

            # Convert states_vectors (and tuples into strings)
            vectors = {
                um.tuples_to_string(key): [vector.tolist() for vector in vectors] for key, vectors in
                self.states_vectors.items()
            }

            # Convert initial state
            initial_state = um.tuples_to_string(self.environment.initial_state)

        else:
            # Convert states_vectors
            vectors = {
                key: [vector.tolist() for vector in vectors] for key, vectors in self.states_vectors.items()
            }

            # Convert initial state
            initial_state = self.environment.initial_state

        # Prepare data to dumps
        data = {
            'time': '{}s'.format(total_time),
            'memory': {
                'v_s_0': len(vectors[initial_state]),
                'full': sum(len(vectors) for vectors in vectors.values())
            },
            'vectors': vectors
        }

        # Extract timestamp
        timestamp = int(time.time())

        # Get environment name in snake case
        env_name = um.str_to_snake_case(self.environment.__class__.__name__)

        # Get only first letter of each word
        env_name_abbr = ''.join([word[0] for word in env_name.split('_')])

        # Specify full path
        if self.limited_precision:
            agent_path = 'b_lp/train_data/{}_{}_{}.yml'.format(env_name_abbr, Vector.decimal_precision, timestamp)
        else:
            agent_path = 'b/train_data/{}_{}.yml'.format(env_name_abbr, timestamp)

        file_path = dumps_path.joinpath(agent_path)

        # If any parents doesn't exist, make it.
        file_path.parent.mkdir(parents=True, exist_ok=True)

        with file_path.open(mode='w+', encoding='UTF-8') as f:

            if self.dumps_format == 'yml':
                f.write(um.structures_to_yaml(data=data))
            else:
                f.write(json.dumps(data))
