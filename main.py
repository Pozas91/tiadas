"""
Main file to test all developed models.
"""
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

import utils.hypervolume as uh
from agents import AgentPQL, AgentMOSP, AgentQ
# from gym_tiadas.envs import *
from gym_tiadas.gym_tiadas.envs import *
from models import Vector
from utils import pareto, q_learning

ENV_NAME_MESH = 'russell-norvig-v0'
ENV_NAME_DISCRETE = 'russell-norvig-discrete-v0'


def plot_training_from_zero():
    # Default reward
    reward = -0.04

    # Environments
    environment_mesh = gym.make(ENV_NAME_MESH, default_reward=reward)

    # Times
    time_train_mesh = list()
    epochs_list = [1000, 10000, 100000, 200000]

    # Agents to show policy at end
    agent_mesh = None

    print("Training agent...")

    for epochs in epochs_list:
        # Training mesh agent
        agent_mesh = AgentQ(environment=environment_mesh)
        start_time = time.time()
        q_learning.train(agent=agent_mesh, epochs=epochs)
        time_train = time.time() - start_time
        time_train_mesh.append(time_train)
        print('MESH: To {} epochs -> {} seconds.'.format(epochs, time_train))

    plt.plot(epochs_list, time_train_mesh, label='Mesh Agent')

    plt.xlabel('# epochs')
    plt.ylabel('Time in seconds')

    plt.legend(loc='upper left')

    plt.show()

    # Best policies
    print('Best policy (Agent mesh)')
    agent_mesh.show_policy()


def plot_training_accumulate():
    # Default reward
    reward = -0.04

    # Environments
    environment_mesh = gym.make(ENV_NAME_MESH, default_reward=reward)

    # Times
    time_train_mesh = list()
    epochs_list = [1000, 10000]

    # Agents to show policy at end
    agent_mesh = AgentQ(environment=environment_mesh, states_to_observe=[(0, 0), (2, 0), (2, 1), (3, 2)])

    print("Training agent...")

    for epochs in epochs_list:
        # Training mesh agent
        start_time = time.time()
        q_learning.train(agent=agent_mesh, epochs=epochs)
        time_train = time.time() - start_time
        time_train_mesh.append(time_train)
        print('MESH: To {} epochs -> {} seconds.'.format(epochs, time_train))

    plt.plot(epochs_list, time_train_mesh, label='Mesh Agent')

    plt.xlabel('# epochs')
    plt.ylabel('Time in seconds')

    plt.legend(loc='upper left')

    plt.show()

    # Best policies
    print('Best policy (Agent mesh)')
    agent_mesh.show_policy()


def plot_performance(epochs=100000):
    # Default reward
    reward = -0.04

    # Environments
    environment_mesh = gym.make(ENV_NAME_MESH, default_reward=reward)

    # Agents to show policy at end
    agent_mesh = AgentQ(environment=environment_mesh, states_to_observe=[(0, 0), (1, 0), (2, 0), (2, 1), (3, 2)],
                        alpha=0.01)

    # Training mesh agent
    print("Training agent...")
    q_learning.train(agent=agent_mesh, epochs=epochs)
    print('Training finished!')

    for state, data in agent_mesh.states_to_observe.items():
        plt.plot(data, label='State: {}'.format(state))

    plt.xlabel('Iterations')
    plt.ylabel('V max')

    plt.legend(loc='upper left')

    plt.show()

    # Best policies
    print('Best policy (Agent mesh)')
    agent_mesh.show_policy()


def russel_and_norvig():
    env = RussellNorvig()
    agent = AgentQ(environment=env)
    q_learning.train(agent=agent, epochs=int(1e5))
    agent.show_policy()
    pass


def deep_sea_treasure():
    env = DeepSeaTreasure()
    weights = (0., 1.)
    agent = AgentMOSP(environment=env, weights=weights, epsilon=0.8, max_iterations=1000)
    q_learning.train(agent=agent)
    agent.show_policy()
    pass


def bonus_world():
    env = BonusWorld()
    weights = (0., 1., 0.)
    agent = AgentMOSP(environment=env, weights=weights, epsilon=0.5, alpha=0.2, states_to_observe=[(0, 0)])
    q_learning.train(agent=agent)
    agent.show_policy()
    pass


def space_exploration():
    env = SpaceExploration()
    weights = (0.8, 0.2)
    agent = AgentMOSP(environment=env, weights=weights, epsilon=0.5, alpha=0.2, states_to_observe=[(0, 0)])
    q_learning.train(agent=agent, epochs=int(1e5))
    agent.show_policy()
    pass


def testing_pareto():
    # Training mesh agent
    start_time = time.time()

    # Build environment
    env = DeepSeaTreasureSimplified()

    # Pareto's points
    pareto_points = env.pareto_optimal

    # Build agent
    weights = (0.99, 0.01)
    agent = AgentMOSP(environment=env, weights=weights, states_to_observe=[(0, 0)], epsilon=0.5, alpha=0.2)

    # Search one extreme objective.
    objective = agent.process_reward(pareto_points[0])
    q_learning.objective_training(agent=agent, objective=objective)

    # Get p point from agent test.
    p = q_learning.testing(agent=agent)

    # Reset agent to train again with others weights
    agent.reset()

    # Set weights to find another extreme point
    agent.weights = [0.01, 0.99]

    # Search the other extreme objective.
    objective = agent.process_reward(pareto_points[-1])
    q_learning.objective_training(agent=agent, objective=objective)

    # Get q point from agent test.
    q = q_learning.testing(agent=agent)

    # Search pareto points
    pareto_frontier = pareto.calc_frontier_scalarized(p=p, q=q, agent=agent, solutions_known=pareto_points)
    pareto_frontier_np = np.array(pareto_frontier)

    # Calc rest of time
    time_train = time.time() - start_time

    # Get pareto point's x axis
    x = pareto_frontier_np[:, 0]

    # Get pareto point's y axis
    y = pareto_frontier_np[:, 1]

    # Build and show plot.
    plt.scatter(x, y)
    plt.ylabel('Reward')
    plt.xlabel('Time')
    plt.show()

    print('Pareto: {} seconds.'.format(time_train))
    pass


def resource_gathering():
    env = ResourceGathering()
    weights = (0., 0., 1.)
    agent = AgentMOSP(environment=env, weights=weights, max_iterations=1000)
    q_learning.train(agent=agent)
    agent.show_policy()
    pass


def pressurized_bountiful_sea_treasure():
    env = PressurizedBountifulSeaTreasure()
    weights = (1., 0., 0.)
    agent = AgentMOSP(environment=env, weights=weights, epsilon=0.5, max_iterations=1000)
    q_learning.train(agent=agent, epochs=int(2e5))
    agent.show_policy()
    pass


def buridan_ass():
    env = BuridanAss()
    weights = (0.3,) * 3
    agent = AgentMOSP(environment=env, epsilon=0.3, weights=weights)
    q_learning.train(agent=agent, epochs=int(1e4))
    pass


def mo_puddle_world():
    env = MoPuddleWorld()
    weights = (0.5,) * 2
    agent = AgentMOSP(environment=env, weights=weights, epsilon=0.3, max_iterations=100)
    q_learning.train(agent=agent, epochs=int(1e3))
    agent.show_policy()
    pass


def linked_rings():
    env = LinkedRings()
    weights = (0.5,) * 2
    agent = AgentMOSP(environment=env, weights=weights, epsilon=0.1, max_iterations=100,
                      states_to_observe=[0, 1, 4])
    q_learning.train(agent=agent, epochs=int(1e3))
    agent.show_raw_policy()
    agent.show_observed_states()
    pass


def non_recurrent_rings():
    env = NonRecurrentRings()
    weights = (0.3, 0.7)
    agent = AgentMOSP(environment=env, weights=weights, epsilon=0.1, max_iterations=100,
                      states_to_observe=[0, 7])
    q_learning.train(agent=agent, epochs=int(1e3))
    agent.show_raw_policy()
    agent.show_observed_states()
    pass


def deep_sea_treasure_simplified():
    env = DeepSeaTreasureSimplified()
    weights = (0.5,) * 2
    agent = AgentMOSP(environment=env, weights=weights, epsilon=0.5, states_to_observe=[(0, 0)])
    q_learning.train(agent=agent, epochs=int(1e4))
    agent.show_observed_states()
    agent.show_policy()
    pass


def deep_sea_treasure_simplified_mo_mp():
    env = DeepSeaTreasure()
    agent = AgentPQL(environment=env, epsilon=0.7, states_to_observe=[(0, 0)], hv_reference=Vector([-25, 0]),
                     evaluation_mechanism='PO-PQL')

    graph = list()

    for _ in range(4):
        agent.reset()
        q_learning.train(agent=agent, epochs=3000)
        graph.append(agent.states_to_observe.get((0, 0)))

    data = np.average(graph, axis=0)
    error = np.std(graph, axis=0)
    y = np.arange(0, len(data), 1)

    plt.errorbar(x=y, y=data, yerr=error, errorevery=500, label='First')
    plt.errorbar(x=y, y=data, yerr=error, errorevery=500, label='Second')
    # plt.plot(data)

    plt.xlabel('Iterations')
    plt.ylabel('HV max')
    plt.legend(loc='upper left')
    plt.show()

    pass


def graphs_dps():
    env = DeepSeaTreasure()
    # If epsilon is greater than 0.7 program may throw an exception with reference point is invalid. (Appears a
    # vector (-26, 124))
    epsilon = 0.70
    states_to_observe = [
        (0, 0)
    ]
    hv_reference = Vector([-25, 0])

    evaluation_mechanisms = ['HV-PQL', 'PO-PQL', 'C-PQL']
    epochs = 3000
    iterations = 50

    # Make instance of AgentPQL
    agent = AgentPQL(environment=env, epsilon=epsilon, states_to_observe=states_to_observe,
                     hv_reference=hv_reference)

    if hasattr(env, 'pareto_optimal'):
        # Get pareto optimal
        pareto_optimal = env.pareto_optimal

        # Calc hypervolume from pareto_optimal and reference vector given
        optimal = uh.calc_hypervolume(list_of_vectors=pareto_optimal, reference=hv_reference)

        # Prepare data to use in graph
        graph = [[optimal] * epochs]
        data = np.average(graph, axis=0)
        error = np.std(graph, axis=0)
        y = np.arange(0, epochs, 1)

        plt.errorbar(x=y, y=data, yerr=error, errorevery=epochs * 0.1, label='Pareto frontier')

    for evaluation_mechanism in evaluation_mechanisms:

        # Reset graph data
        graph = list()

        # Set evaluation mechanism
        agent.evaluation_mechanism = evaluation_mechanism

        for i in range(iterations):
            # Reset agent
            agent.reset()

            # Train model
            q_learning.train(agent=agent, epochs=epochs)

            # Get graph data
            graph.append(agent.states_to_observe.get(states_to_observe[0]))

        # Prepare data to graph
        data = np.average(graph, axis=0)
        error = np.std(graph, axis=0)
        y = np.arange(0, epochs, 1)

        # Set graph to current evaluation mechanism
        plt.errorbar(x=y, y=data, yerr=error, errorevery=epochs * 0.1, label=evaluation_mechanism)

    # Show data
    plt.xlabel('Iterations')
    plt.ylabel('HV max')
    plt.legend(loc='upper left')
    plt.show()


def track_policy():
    # Settings variables

    # class name of the environment
    env = DeepSeaTreasure()
    evaluation_mechanism = 'C-PQL'

    # Try to load last model
    agent = AgentPQL.load(environment=env, evaluation_mechanism=evaluation_mechanism)

    if agent is None:
        # Get instance of agent
        agent = AgentPQL(environment=env, epsilon=0.7, states_to_observe=[(0, 0)],
                         hv_reference=Vector([-25, 0]), evaluation_mechanism=evaluation_mechanism)

        # Train model
        q_learning.train(agent=agent, epochs=3000)

        # Save model
        agent.save()

    # else:
    #     # Train model
    #     q_learning.train(agent=agent, epochs=18000)
    #
    #     # Save model
    #     agent.save()

    # state = (5, 2)  # for space exploration
    # Another environments
    state = (0, 0)
    target = Vector([-8, 15])

    non_dominated_vectors = agent.non_dominated_vectors_from_state(state=state)
    agent.show_observed_states()
    path = agent.track_policy(state=state, target=target)

    agent.print_information()
    print("Target: {}".format(target))
    print("Pareto's vectors: {}".format(non_dominated_vectors))
    print("Found path: {}".format(path))
    pass


def main():
    # plot_training_from_zero()
    # plot_training_accumulate()
    # plot_performance(epochs=1000000)

    # deep_sea_treasure()
    # testing_pareto()
    # bonus_world()
    # resource_gathering()
    # pressurized_bountiful_sea_treasure()
    # buridan_ass()
    # mo_puddle_world()

    # space_exploration()

    # linked_rings()
    # non_recurrent_rings()
    # russel_and_norvig()
    # deep_sea_treasure_simplified()
    # deep_sea_treasure_simplified_mo_mp()
    track_policy()
    # graphs_dps()
    pass


if __name__ == '__main__':
    main()
