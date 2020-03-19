"""
AgentPQLExp3.

A modification of AgentPQLEXP where selection probabilities are such that with epsilon = 0.5 we obtain a random walk.
In order to ensure that epsilon = 0.5 yields a random walk, laplacian smoothing of selection probabilities is
carried out.

"""

import numpy as np

from . import AgentPQLEXP


class AgentPQLEXP3(AgentPQLEXP):

    def _best_action(self, state: object = None, extra: object = None) -> int:
        """
        Select action proportional to the credit indicated in train_data. If necessary, selection probabilities are
        smoothed.
        train_data is a tuple (maximum_credit, list of tuples: (action, credit), sum_credit)

         Let E be the maximum credit, N the number of actions, and S the sum of all credits.
        If E > 2S/N, then laplacian smoothing is carried out (otherwise, it is not always possible to calculate
        the

        :param extra:
        :param state:
        :return:
        """
        n = self.environment.action_space.n

        e = extra[0]
        info = extra[1]
        s = extra[2]

        # Amount to add to smooth probabilities
        k = max(0, (e - ((2 * s) / n)))

        # Calculate array of accumulated credit
        accumulation = np.zeros(n)
        summation = 0  # At end should be state + nk

        for i in range(n):
            summation += info[i][1] + k
            accumulation[i] = summation

        if summation == 0:
            return self.environment.action_space.sample()

        # Select action with probability proportional to hv
        random_number = self.generator.uniform(low=0, high=summation)

        for i in range(n):
            if random_number <= accumulation[i]:
                return info[i][0]

        print('Warning: agent_pql_exp._best_action: Selecting emergency action')
        return info[n - 1][0]

    def _non_greedy_action(self, state: object = None, extra: object = None) -> int:
        """
        Select action with probability inversely proportional to the credit indicated in extra.
        If necessary, probabilities are smoothed so that with epsilon = 0.5, a random walk is obtained.
        train_data is a  tuple. The first element is the maximum credit, the second a list of tuples: (action, credit),
        and the third the sum of credits.

        :param state:
        :param extra:
        :return:
        """

        n = self.environment.action_space.n

        e = extra[0]
        info = extra[1]
        s = extra[2]

        # Amount to add to smooth probabilities
        k = max(0, (e - ((2 * s) / n)))

        # Sum of smooths credits
        s2 = s + n * k

        # Calculate array accumulated probabilities
        acu = np.zeros(n)

        if s == 0:
            return self.environment.action_space.sample()

        # Calculate probabilities inversely proportional to score
        ns2 = s2 * n
        summation = 0

        for i in range(n):
            aux = (2 * s2 - n * (info[i][1] + k))
            # (s2 - nact * extra[i][1])
            summation += aux
            acu[i] = summation

        # Select action with probability proportional to hv
        num = self.generator.uniform(low=0, high=ns2)

        for i in range(n):
            if num <= acu[i]:
                return info[i][0]

        print('Warning: agent_pql_exp2._non_greedy_action: Selecting emergency action')
        return info[n - 1][0]
