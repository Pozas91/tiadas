from agents.agent_bn import AgentBN
from environments import ResourceGathering
from policies import rg_policies


def main():
    # Build agent
    agent: AgentBN = AgentBN(environment=ResourceGathering(), gamma=0.9)

    # Policies
    policies = rg_policies.copy()

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


if __name__ == '__main__':
    # main()
    main()
