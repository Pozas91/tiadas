"""
Main file to test all develop models.
"""
import time

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym_tiadas.envs import *

from agents import Agent, AgentMultiObjective
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


def deep_sea_treasure():
    environment = DeepSeaTreasure()
    agent = AgentMultiObjective(environment=environment, weights=[0., 1.], epsilon=0.8, max_iterations=1000)
    q_learning.train(agent=agent, verbose=True)
    agent.show_policy()
    pass


def testing_pareto():
    # Build environment
    env = DeepSeaTreasureSimplified()

    # Build agent
    agent = AgentMultiObjective(environment=env, weights=[0.9, 0.1], states_to_observe=[(0, 0)], epsilon=0.5,
                                alpha=0.1, gamma=1.)

    t0 = time.time()
    q_learning.cheat_train(agent=agent, objective=-0.4, close_margin=1e-1)
    time_train = time.time() - t0
    print('Time train: {:.2f} seconds.'.format(time_train))

    agent.show_policy()

    # p = agent.v
    # p = tuple(np.multiply(agent.weights, p))
    p = tuple(q_learning.testing(agent=agent))

    # Reset agent
    agent.reset()

    agent.set_rewards_weights([0.1, 0.9])
    t0 = time.time()
    # q_learning.cheat_train(agent=agent, objective=107.5, close_margin=1e-1)
    q_learning.train(agent=agent)
    time_train = time.time() - t0
    print('Time train: {:.2f} seconds.'.format(time_train))
    agent.show_policy()

    q = tuple(q_learning.testing(agent=agent))

    # q = agent.v
    # q = tuple(np.multiply(agent.weights, q))

    pareto_frontier = pareto.algorithm(p=p, q=q, problem=agent)
    pareto_frontier_np = np.array(pareto_frontier)

    x = pareto_frontier_np[:, 0]
    y = pareto_frontier_np[:, 1]

    plt.scatter(x, y)
    plt.ylabel('Reward')
    plt.xlabel('Time')
    plt.show()


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


def main():
    # plot_training_from_zero()
    # plot_training_accumulate()
    # plot_performance(epochs=1000000)

    # deep_sea_treasure()
    testing_pareto()
    # resource_gathering()
    # pressurized_bountiful_sea_treasure()
    # buridan_ass()
    # mo_puddle_world()
    pass


if __name__ == '__main__':
    main()
