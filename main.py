import gym
import gym_tiadas
import time
import matplotlib.pyplot as plt

from agents import Agent, AgentDiscrete, AgentMultiObjective
from gym_tiadas.envs import BuridanAss, DeepSeaTreasure, ResourceGathering, PressurizedBountifulSeaTreasure, \
    MoPuddleWorld

# ENV_NAME = 'CartPole-v1'
ENV_NAME_MESH = 'russell-norvig-v0'
ENV_NAME_DISCRETE = 'russell-norvig-discrete-v0'


def training(agent, epochs=100000, verbose=False):
    """
    Return an agent trained with `epochs` epochs.
    :param verbose:
    :param agent:
    :param epochs:
    :return:
    """

    for _ in range(epochs):
        # Do an episode
        agent.episode()

    # Show values
    if verbose:
        agent.show_crude_policy()
        agent.show_q()


def plot_training_from_zero():
    # Default reward
    reward = -0.04

    # Environments
    environment_mesh = gym.make(ENV_NAME_MESH, default_reward=reward)
    environment_discrete = gym.make(ENV_NAME_DISCRETE, default_reward=reward)

    # Times
    time_train_mesh = list()
    time_train_discrete = list()
    epochs_list = [1000, 10000, 100000, 200000]

    # Agents to show policy at end
    agent_mesh = None
    agent_discrete = None

    print("Training agent...")

    for epochs in epochs_list:
        # Training mesh agent
        agent_mesh = Agent(environment=environment_mesh)
        start_time = time.time()
        training(agent=agent_mesh, epochs=epochs)
        time_train = time.time() - start_time
        time_train_mesh.append(time_train)
        print('MESH: To {} epochs -> {} seconds.'.format(epochs, time_train))

        # Training discrete agent
        agent_discrete = AgentDiscrete(environment=environment_discrete)
        start_time = time.time()
        training(agent=agent_discrete, epochs=epochs)
        time_train = time.time() - start_time
        time_train_discrete.append(time_train)
        print('DISCRETE: To {} epochs -> {} seconds.'.format(epochs, time_train))

        print()

    plt.plot(epochs_list, time_train_mesh, label='Mesh Agent')
    plt.plot(epochs_list, time_train_discrete, label='Discrete Agent')

    plt.xlabel('# epochs')
    plt.ylabel('Time in seconds')

    plt.legend(loc='upper left')

    plt.show()

    # Best policies
    print('Best policy (Agent mesh)')
    agent_mesh.show_policy()

    print('Best policy (Agent discrete)')
    agent_discrete.show_policy()


def plot_training_accumulate():
    # Default reward
    reward = -0.04

    # Environments
    environment_mesh = gym.make(ENV_NAME_MESH, default_reward=reward)
    environment_discrete = gym.make(ENV_NAME_DISCRETE, default_reward=reward)

    # Times
    time_train_mesh = list()
    time_train_discrete = list()
    epochs_list = [1000, 10000]

    # Agents to show policy at end
    agent_mesh = Agent(environment=environment_mesh, states_to_observe=[(0, 0), (2, 0), (2, 1), (3, 2)])
    agent_discrete = AgentDiscrete(environment=environment_discrete)

    print("Training agent...")

    for epochs in epochs_list:
        # Training mesh agent
        start_time = time.time()
        training(agent=agent_mesh, epochs=epochs)
        time_train = time.time() - start_time
        time_train_mesh.append(time_train)
        print('MESH: To {} epochs -> {} seconds.'.format(epochs, time_train))

        # Training discrete agent
        start_time = time.time()
        training(agent=agent_discrete, epochs=epochs)
        time_train = time.time() - start_time
        time_train_discrete.append(time_train)
        print('DISCRETE: To {} epochs -> {} seconds.'.format(epochs, time_train))

        print()

    plt.plot(epochs_list, time_train_mesh, label='Mesh Agent')
    plt.plot(epochs_list, time_train_discrete, label='Discrete Agent')

    plt.xlabel('# epochs')
    plt.ylabel('Time in seconds')

    plt.legend(loc='upper left')

    plt.show()

    # Best policies
    print('Best policy (Agent mesh)')
    agent_mesh.show_policy()

    print('Best policy (Agent discrete)')
    agent_discrete.show_policy()


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
    training(agent=agent_mesh, epochs=epochs)
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
    agent = AgentMultiObjective(environment=environment, rewards_weights=[.5, 1.], epsilon=0.7)
    training(agent=agent, verbose=True)
    agent.show_policy()
    pass


def resource_gathering():
    environment = ResourceGathering()
    agent = AgentMultiObjective(environment=environment, rewards_weights=[0., 0., .1], max_iterations=1000)
    training(agent=agent, verbose=True)
    pass


def pressurized_bountiful_sea_treasure():
    environment = PressurizedBountifulSeaTreasure()
    agent = AgentMultiObjective(environment=environment, rewards_weights=[1., 0., 0.], epsilon=0.5)
    training(agent=agent, epochs=200000, verbose=True)
    agent.show_policy()
    pass


def buridan_ass():
    environment = BuridanAss()
    agent = AgentMultiObjective(environment=environment, epsilon=0.3, rewards_weights=[0.3, 0.3, 0.3])
    training(agent=agent, epochs=10000, verbose=True)
    pass


def mo_puddle_world():
    environment = MoPuddleWorld()
    agent = AgentMultiObjective(environment=environment, rewards_weights=[0.5, 0.5], epsilon=0.3, max_iterations=100)
    training(agent=agent, epochs=1000, verbose=True)
    agent.show_policy()
    pass


def main():
    # plot_training_from_zero()
    # plot_training_accumulate()
    # plot_performance(epochs=1000000)

    # deep_sea_treasure()
    # resource_gathering()
    # pressurized_bountiful_sea_treasure()
    # buridan_ass()
    mo_puddle_world()
    pass


if __name__ == '__main__':
    main()
