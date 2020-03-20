from agents import Agent, AgentMPQ
from agents.agent_bn import AgentBN
from environments import ResourceGatheringEpisodic, ResourceGathering
from models import GraphType, Vector
from policies import rge_policies


def load_agent() -> Agent:
    # Models path
    # models_path = 'mpq/models/dstrds_1580904518_0.1_2.bin'
    models_path = 'mpq/models/dstrds_1580646055_0.001_5.bin'

    return AgentMPQ.load(filename=models_path)


def train_agent() -> Agent:
    # Environment
    # environment = DeepSeaTreasureRightDownStochastic(columns=3)
    # environment = DeepSeaTreasureRightDown(columns=3)
    # environment = PyramidMDPNoBounces(diagonals=3, n_transition=0.95)
    # environment = DeepSeaTreasure()
    environment = ResourceGathering()

    # Agent
    # agent = AgentMPQ(
    #     environment=environment, hv_reference=environment.hv_reference, alpha=0.1, epsilon=0.4, max_steps=1000
    # )
    # agent = AgentMPQ(environment=environment, hv_reference=environment.hv_reference, alpha=0.01)
    agent = AgentBN(environment=environment, gamma=.9)

    # Vector precision
    Vector.set_decimal_precision(decimal_precision=0.01)

    # Train agent
    # agent.train(graph_type=GraphType.SWEEP, tolerance=0.00001)
    agent.train(graph_type=GraphType.SWEEP, limit=13)

    return agent


def get_trained_agent() -> Agent:
    return train_agent()
    # return load_agent()


def evaluate_predefined_policies():
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
    agent: AgentBN = get_trained_agent()

    # Set initial state
    initial_state = ((2, 4), (0, 0))

    # agent: AgentBN = AgentBN.load(
    #     filename='bn/models/rg_1584437328_0.005.bin'
    # )

    v_s_0 = agent.v[initial_state]
    vectors = Vector.m3_max(set(v_s_0))

    # Simulation
    simulation = dict()

    # Set decimal precision
    Vector.set_decimal_precision(decimal_precision=0.0000001)

    for vector in vectors:
        # Recreate the index objective vector.
        # objective_vector = IndexVector(
        #     index=vector, vector=trained_agent.v[initial_state][vector]
        # )

        objective_vector = vector.copy()

        # Get simulation from this agent
        policy = agent.recover_policy(
            initial_state=initial_state, objective_vector=objective_vector, iterations_limit=agent.total_sweeps
        )

        policy_evaluated = agent.evaluate_policy(policy=policy, tolerance=0.0000001)

        simulation.update({objective_vector: (policy, policy_evaluated)})

    print(simulation)


if __name__ == '__main__':
    main()
    # evaluate_predefined_policies()
