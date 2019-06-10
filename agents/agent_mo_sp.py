"""
Agent multi-objective single-policy.
Convert the rewards vector into a scalarized reward, after that use Q-Learning method. It follow same process that agent
model, but the reward its calculate multiply the weights vector and the rewards vector.

EXAMPLE OF USE OF AgentMOSP:

    # Build environment
    env = DeepSeaTreasureSimplified()

    # Pareto's points
    pareto_points = env.pareto_optimal

    # Build agent
    agent = AgentMOSP(environment=env, weights=[0.99, 0.01], states_to_observe=[(0, 0)], epsilon=0.5, alpha=0.2)

    # Search one extreme objective.
    objective = agent.process_reward(pareto_points[0])
    q_learning.objective_training(agent=agent, objective=objective, close_margin=1e-2)

    # Get p point from agent test.
    p = q_learning.testing(agent=agent)

    # Reset agent to train again with others weights
    agent.reset()

    # Set weights to find another extreme point
    agent.weights = [0.01, 0.99]

    # Search the other extreme objective.
    objective = agent.process_reward(pareto_points[-1])
    q_learning.objective_training(agent=agent, objective=objective, close_margin=1e-1)

    # Get q point from agent test.
    q = q_learning.testing(agent=agent)

    # Search pareto points (THIS IS THE CALL OF THESE FUNCTIONS)
    pareto_frontier = pareto.calc_frontier_scalarized(p=p, q=q, problem=agent, solutions_known=pareto_points)
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
"""
import numpy as np

import utils.miscellaneous as um
from gym_tiadas.gym_tiadas.envs import Environment
from models import Vector, VectorFloat
from .agent_q import AgentQ


class AgentMOSP(AgentQ):

    def __init__(self, environment: Environment, alpha: float = 0.1, epsilon: float = 0.1, gamma: float = 1.,
                 seed: int = 0, states_to_observe: list = None, max_steps: int = None, weights: tuple = None):
        """
        :param environment: An environment where agent does any operation.
        :param alpha: Learning rate
        :param epsilon: Epsilon using in e-greedy policy, to explore more states.
        :param gamma: Discount factor
        :param seed: Seed used for np.random.RandomState method.
        :param states_to_observe: List of states from that we want to get a graphical output.
        :param max_steps: Limits of steps per episode.
        :param weights: Tuple of weights to multiply per reward vector.
        """

        # Sum of weights must be 1.
        assert weights is not None and np.sum(weights) == 1.

        # Super call init
        super().__init__(environment=environment, alpha=alpha, epsilon=epsilon, gamma=gamma, seed=seed,
                         states_to_observe=states_to_observe, max_steps=max_steps)

        # Set weights
        self.weights = weights

    def process_reward(self, reward: Vector) -> Vector:
        """
        Processing reward function.
        :param reward:
        :return:
        """

        # Multiply the reward for the vector weights, sum all components and return a reward of the same type as the
        # original, but with only one component.
        return reward.__class__(float(np.sum(reward * self.weights)))

    def _update_q_values(self, reward: Vector, action: int, next_state: object) -> None:
        """
        Update Q-Dictionary with new data
        :param reward:
        :param action:
        :param next_state:
        :return:
        """

        # Apply function
        reward = self.process_reward(reward=reward)

        # Super call
        super()._update_q_values(reward=reward, action=action, next_state=next_state)

    def find_c_vector(self, w1: float, w2: float, solutions_known: list = None) -> Vector:
        """
        This method is called from calc_frontier_scalarized method.

        Try to find an c point to add to the pareto's frontier. There are two options:
            * We know the solutions of pareto's frontier, and training the agent until get max solution.
            * We don't know the solutions and training to try get max solution.

        :param solutions_known: If we know the possible solutions, we can indicate them to the algorithm to improve the
            training of the agent. If is None, then is ignored.
        :param w1: first weight
        :param w2: second weight
        :return:
        """

        # Reset agent (forget q-values, initial_state, etc.).
        self.reset()

        # Set news weights to get the new solution.
        self.weights = [w1, w2]

        # If solutions not is None
        if solutions_known:
            # Multiply and sum all points with agent's weights.
            objectives = np.sum(np.multiply(solutions_known, [w1, w2]), axis=1)

            # Get max of these sums (That is the objective).
            objective = VectorFloat(np.max(objectives))

            # Train agent searching that objective.
            self.objective_training(objective=objective)
        else:
            # Normal training.
            self.train()

        # Get point c from agent's test.
        c = self.testing()

        return c

    def calc_frontier_scalarized(self, p: Vector, q: Vector, solutions_known: list = None) -> list:
        """
        This is a main method to calc pareto's frontier.

        Return a list of supported solutions costs, this method is only valid to two objectives problems.
        Applies a dichotomous search to find all supported solutions costs.

        :param solutions_known: If we know the possible solutions, we can indicate them to the algorithm to improve the
            training of the agent. If is None, then is ignored.
        :param p: 2D point
        :param q: 2D point
        :return:
        """

        # A new list with p and q
        result = [p, q]

        # Create a new stack
        accumulate = list()

        # Push a vector with p and q in the stack
        accumulate.append(tuple(result))

        while len(accumulate) > 0:
            # Pop the next pair of points from the stack.
            a, b = accumulate.pop()

            # Order points nearest to the center using euclidean distance.
            a, b = tuple(um.order_vectors_by_origin_nearest([a, b]))

            # Decompose points
            a_x, a_y = a
            b_x, b_y = b

            # Calculate the parameters of the new linear objective function (multiply by -1. to convert in maximize
            # problem)
            w1 = np.multiply(a_y - b_y, -1.)
            w2 = np.multiply(b_x - a_x, -1.)

            # Solve P to find a new solution ang get its cost vector c.
            c = self.find_c_vector(w1, w2, solutions_known=solutions_known)

            # Decompose c vector.
            c_x, c_y = c

            if (w1 * a_x + w2 * a_y) != (w1 * c_x + w2 * c_y):
                # c is the cost of a new supported solution

                # Push new pair in the stack
                accumulate.append((a, c))

                # Push new pair in the stack
                accumulate.append((c, b))

                # Add c to the result
                result.append(c)

        return result
