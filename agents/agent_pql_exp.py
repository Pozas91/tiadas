"""
AgentPQLExp.

A modification of AgentPQL that implements more general exploration/exploitation mechanisms in action selection.
Each action selection is assigned a credit (HV,PO,C) according to its contribution of Pareto-optimal policies.
During exploitation (greedy action selection), each action is selected with a probability proportional to its credit.
During exploration, each action is selected with a probability inversely proportional to its credit.

"""

import numpy as np

import utils.hypervolume as uh
from agents.agent_pql import AgentPQL
from models import IndexVector, EvaluationMechanism


class AgentPQLEXP(AgentPQL):

    def select_action(self, state: object = None) -> int:
        """
        Select best action with a little e-greedy policy.
        :return:
        """

        # If state is None, then set current state to state.
        if not state:
            state = self.state

        if self.evaluation_mechanism is EvaluationMechanism.HV:
            data = self.calculate_hypervolume()
        elif self.evaluation_mechanism is EvaluationMechanism.C:
            data = self.calculate_cardinality()
        elif self.evaluation_mechanism is EvaluationMechanism.PO:
            data = self.calculate_pareto()
        elif self.evaluation_mechanism is EvaluationMechanism.CHV:
            data = self.calculate_chv()
        else:
            raise ValueError('Unknown evaluation mechanism')

        if self.generator.uniform(low=0., high=1.) < self.epsilon:
            # Get random action to explore possibilities
            return self._greedy_action(state, data)
        else:
            # Get best action to exploit reward.
            return self.best_action(state, data)

    def best_action(self, state: object = None, data=None) -> int:
        """
        Select action proportional to the credit indicated in info.
        data is a tuple (maximum_credit, list of tuples: (action, credit))
        :param state:
        :param info:
        :return:
        """
        len = self.environment.action_space.n

        info = data[1]

        # calculate array of acumulated credit
        acu = np.zeros(len)
        sum = 0
        for i in range(len):
            sum += info[i][1]
            acu[i] = sum

        if sum == 0:
            # print('Warning: zero credit')
            return self.environment.action_space.sample()

        # select action with probability proportional to hv
        num = self.generator.uniform(low=0, high=sum)

        # print('--AgentPQLEXP-best_action----info-acu-num-accion------')
        # print(info)
        # print(acu)
        # print(num)

        for i in range(len):
            if num <= acu[i]:
                # print(info[i][0])
                return info[i][0]

        print('Warning: agent_pql_exp.best_action: seleccionando acción de emergencia')
        return info[len - 1][0]

    def _greedy_action(self, state: object = None, data=None) -> int:
        """
        Select action proportional to the credit indicated in info.
        data is a  tuple. The first element is the maximum credit, and the sencond a list of tuples: (action, credit)
        :param state:
        :param info:
        :return:
        """
        len = self.environment.action_space.n

        maximum = data[0]
        info = data[1]

        # calculate array of acumulated maximum - credit
        acu = np.zeros(len)
        sum = 0
        for i in range(len):
            sum += (maximum - info[i][1])
            acu[i] = sum

        if sum == 0:
            # print('Warning: zero credit')
            return self.environment.action_space.sample()

        # select action with probability proportional to hv
        num = self.generator.uniform(low=0, high=sum)

        # print('--AgentPQLEXP- greedy_action----info-acu-num-accion------')
        # print(info)
        # print(acu)
        # print(num)

        for i in range(len):
            if num <= acu[i]:
                # print(info[i][0])
                return info[i][0]

        print('Warning: agent_pql_exp._greedy_action: seleccionando acción de emergencia')
        return info[len - 1][0]

    def calculate_hypervolume(self):
        """
        Calc the hypervolume for each action and returns a list of tuples (maximum-hv, [(action, hipervolume)*], sum-hv)
        :param state:
        :return:
        """

        result = list()

        maximum = float('-inf')
        sum = 0
        # for each a in actions
        for a in self.environment.action_space:

            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=self.state, action=a)

            # Calc hypervolume of Q_set, with reference given, and store in list with action
            hv = uh.calc_hypervolume(list_of_vectors=q_set, reference=self.hv_reference)
            result.append((a, hv))
            if hv > maximum:
                maximum = hv

            sum += hv

        # return (max-hv, list of tuples, sum-hv).
        return maximum, result, sum

    def calculate_cardinality(self):
        """
        Calc the cardinality for each action and returns a tuple
        (maximum_cardinality, list of tuples (action, cardinality), sum of cardinalities)

        CAUTION: This method assumes actions are integers in a range.

        :param state:
        :return:
        """

        # List of all Qs
        all_q = list()

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=self.state, action=a)

            # for each Q in Q_set(s, a)
            for q in q_set:
                all_q.append(IndexVector(index=a, vector=q))

        # NDQs <- ND(all_q). Keep only the non-dominating solutions
        actions = IndexVector.actions_occurrences_based_m3_with_repetitions(
            vectors=all_q, actions=action_space
        )

        result = []
        maximum = -1
        summation = 0

        for a in action_space:
            result.append((a, actions[a]))
            if actions[a] > maximum:
                maximum = actions[a]
            summation += actions[a]

        return maximum, result, summation

    def calculate_pareto(self):
        """
        Calc if a Pareto estimate exists for each action and returns a
        tuple (1, list of tuples (action, 1-exists, 0-does-not-exist), sum of 1s)

        CAUTION: This method assumes actions are integers in a range.

        :param state:
        :return:
        """

        # List of all Qs
        all_q = list()

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=self.state, action=a)

            # for each Q in Q_set(s, a)
            for q in q_set:
                all_q.append(IndexVector(index=a, vector=q))

        # NDQs <- ND(all_q). Keep only the non-dominating solutions
        actions = IndexVector.actions_occurrences_based_m3_with_repetitions(
            vectors=all_q, actions=action_space
        )

        result = []
        summation = 0
        for a in action_space:
            if actions[a] > 0:
                result.append((a, 1))
                summation += 1
            else:
                result.append((a, 0))

        return 1, result, summation

    def calculate_chv(self):
        """
        Calc the hypervolume for the vectors that provide cardinality for each action and returns a tuple
        (maximum_chv, list of tuples (action, chv), sum of chv)

        CAUTION: This method assumes actions are integers in a range.

        :param state:
        :return:
        """

        # List of all Qs
        all_q = list()

        # Getting action_space
        action_space = self.environment.action_space

        # for each a in actions
        for a in action_space:

            # Get Q-set from state given for each possible action.
            q_set = self.q_set(state=self.state, action=a)

            # for each Q in Q_set(s, a)
            for q in q_set:
                all_q.append(IndexVector(index=a, vector=q))

        # NDQs <- ND(all_q). Keep only the non-dominating solutions (We want the vectors, so return_vectors must be
        # True)
        vectors = IndexVector.actions_occurrences_based_m3_with_repetitions(
            vectors=all_q, actions=action_space, returns_vectors=True
        )

        result = []
        maximum = -1
        summation = 0

        for a in action_space:

            chv = 0

            if len(vectors[a]) > 0:
                chv = uh.calc_hypervolume(list_of_vectors=vectors[a], reference=self.hv_reference)

            result.append((a, chv))
            maximum = max(maximum, chv)
            summation += chv

        return maximum, result, summation
