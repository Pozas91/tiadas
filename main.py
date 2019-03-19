"""
Main file to test all develop models.
"""
import time

import gym
import matplotlib.pyplot as plt
import numpy as np

from agents import Agent, AgentMultiObjective
# from gym_tiadas.envs import *
from gym_tiadas.gym_tiadas.envs import *
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
        agent_mesh = Agent(environment=environment_mesh)
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
    agent_mesh = Agent(environment=environment_mesh, states_to_observe=[(0, 0), (2, 0), (2, 1), (3, 2)])

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
    agent_mesh = Agent(environment=environment_mesh, states_to_observe=[(0, 0), (1, 0), (2, 0), (2, 1), (3, 2)],
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
    environment = RussellNorvig()
    agent = Agent(environment=environment)
    q_learning.train(agent=agent, verbose=True, epochs=int(1e5))
    agent.show_policy()
    pass


def deep_sea_treasure():
    environment = DeepSeaTreasure()
    agent = AgentMultiObjective(environment=environment, weights=[0., 1.], epsilon=0.8, max_iterations=1000)
    q_learning.train(agent=agent, verbose=True)
    agent.show_policy()
    pass


def bonus_world():
    environment = BonusWorld()
    agent = AgentMultiObjective(environment=environment, weights=[0., 1., 0.], epsilon=0.5, alpha=0.2, gamma=1.,
                                states_to_observe=[(0, 0)])
    q_learning.train(agent=agent, verbose=True)
    agent.show_policy()
    pass


def space_exploration():
    environment = SpaceExploration()
    agent = AgentMultiObjective(environment=environment, weights=[0.8, 0.2], epsilon=0.5, alpha=0.2, gamma=1.,
                                states_to_observe=[(0, 0)])
    q_learning.train(agent=agent, verbose=True, epochs=int(1e5))
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
    agent = AgentMultiObjective(environment=env, weights=[0.99, 0.01], states_to_observe=[(0, 0)],
                                epsilon=0.5, alpha=0.2)

    # Search one extreme objective.
    objective = agent.process_reward(pareto_points[0])
    q_learning.objective_training(agent=agent, objective=objective, close_margin=1e-2)

    # Get p point from agent test.
    p = q_learning.testing(agent=agent)

    # Reset agent
    agent.reset()

    # Set weights to find another extreme point
    agent.weights = [0.01, 0.99]

    # Search the other extreme objective.
    objective = agent.process_reward(pareto_points[-1])
    q_learning.objective_training(agent=agent, objective=objective, close_margin=1e-1)

    # Get q point from agent test.
    q = q_learning.testing(agent=agent)

    # Search pareto points
    pareto_frontier = pareto.calc_frontier(p=p, q=q, problem=agent, solutions_known=pareto_points)
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
    environment = ResourceGathering()
    agent = AgentMultiObjective(environment=environment, weights=[0., 0., .1], max_iterations=1000)
    q_learning.train(agent=agent, verbose=True)
    agent.show_policy()
    pass


def pressurized_bountiful_sea_treasure():
    environment = PressurizedBountifulSeaTreasure()
    agent = AgentMultiObjective(environment=environment, weights=[1., 0., 0.], epsilon=0.5, max_iterations=1000)
    q_learning.train(agent=agent, epochs=200000, verbose=True)
    agent.show_policy()
    pass


def buridan_ass():
    environment = BuridanAss()
    agent = AgentMultiObjective(environment=environment, epsilon=0.3, weights=[0.3, 0.3, 0.3])
    q_learning.train(agent=agent, epochs=10000, verbose=True)
    pass


def mo_puddle_world():
    environment = MoPuddleWorld()
    agent = AgentMultiObjective(environment=environment, weights=[0.5, 0.5], epsilon=0.3, max_iterations=100)
    q_learning.train(agent=agent, epochs=1000, verbose=True)
    agent.show_policy()
    pass


def linked_rings():
    environment = LinkedRings()
    agent = AgentMultiObjective(environment=environment, weights=[0.5, 0.5], epsilon=0.1, max_iterations=100,
                                states_to_observe=[0, 1, 4])
    q_learning.train(agent=agent, epochs=int(1e3), verbose=True)
    agent.show_raw_policy()
    agent.print_observed_states()
    pass


def non_recurrent_rings():
    environment = NonRecurrentRings()
    agent = AgentMultiObjective(environment=environment, weights=[0.3, 0.7], epsilon=0.1, max_iterations=100,
                                states_to_observe=[0, 7])
    q_learning.train(agent=agent, epochs=int(1e3), verbose=True)
    agent.show_raw_policy()
    agent.print_observed_states()
    pass


def deep_sea_treasure_simplified():
    environment = DeepSeaTreasureSimplified()
    weights = [0.5] * 2
    agent = AgentMultiObjective(environment=environment, weights=weights, epsilon=0.5,
                                states_to_observe=[(0, 0)])
    q_learning.train(agent=agent, epochs=int(1e4), verbose=True)
    agent.print_observed_states()
    agent.show_policy()
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
    pass


if __name__ == '__main__':
    main()
