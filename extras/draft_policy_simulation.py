"""
Example of train, recover policies, evaluating policies, and evaluating predefined policies.
"""
from agents.agent_bn import AgentBN
from environments import ResourceGatheringEpisodic
from models import GraphType, Vector
from policies import rge_policies


def get_trained_agent() -> AgentBN:
    # Environment
    environment = ResourceGatheringEpisodic()

    # Agent
    agent = AgentBN(environment=environment, gamma=.9)

    # Vector precision
    Vector.set_decimal_precision(decimal_precision=0.01)

    # Train agent
    agent.train(graph_type=GraphType.SWEEP, limit=30)

    return agent


def evaluate_predefined_policies():
    """
    Evaluate manually predefined policies
    :return:
    """
    # Build agent
    agent: AgentBN = AgentBN(environment=ResourceGatheringEpisodic(), gamma=0.9)

    # Policies
    policies = rge_policies.copy()

    # Simulation
    simulation = dict()

    for n, policy in enumerate(policies):
        # # of policy
        n += 1

        # Evaluate policy
        policy_evaluated = agent.evaluate_policy(policy=policy, tolerance=0.000001)

        # Update simulation
        simulation.update({
            n: policy_evaluated
        })

    print(simulation)


def main():
    # Get trained agent
    print('Training agent...')
    agent: AgentBN = get_trained_agent()

    # Set initial state
    initial_state = ((2, 4), (0, 0), False)

    # Initial vectors
    v_s_0 = agent.v[initial_state]
    vectors = Vector.m3_max(set(v_s_0))

    # Show information
    print('Vectors obtained after m3_max algorithm: ')
    print(vectors, end='\n\n')

    # Define a tolerance
    decimal_precision = 0.0000001

    # Simulation
    simulation = dict()

    # Set decimal precision
    Vector.set_decimal_precision(decimal_precision=decimal_precision)

    print('Evaluating policies gotten...')

    # For each vector
    for vector in vectors:
        # Specify objective vector
        objective_vector = vector.copy()

        print('Recovering policy for objective vector: {}...'.format(objective_vector))

        # Get simulation from this agent
        policy = agent.recover_policy(
            initial_state=initial_state, objective_vector=objective_vector, iterations_limit=agent.total_sweeps
        )

        print('Evaluating policy obtaining...', end='\n\n')

        # Train until converge with `decimal_precision` tolerance.
        policy_evaluated = agent.evaluate_policy(policy=policy, tolerance=decimal_precision)

        # Save policy and it evaluation.
        simulation.update({objective_vector: (policy, policy_evaluated)})

    print(simulation)


if __name__ == '__main__':
    main()
    # evaluate_predefined_policies()
