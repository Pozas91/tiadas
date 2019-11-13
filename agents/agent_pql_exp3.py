"""
AgentPQLExp3.

A modification of AgentPQLEXP where selection probabilities are such that with epsilon = 0.5 we obtain a random walk.
In order to ensure that epsilon = 0.5 yields a random walk, laplacian smoothing of selection probabilities is
carried out.

"""

import numpy as np

from agents.agent_pql_exp import AgentPQLEXP


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

        # cantidad a añadir para suavizar las probabilidades
        k = max(0, (e - ((2 * s) / n)))

        # calculate array of accumulated credit
        accumulation = np.zeros(n)
        summation = 0  # al final debería ser s + nk
        for i in range(n):
            summation += info[i][1] + k
            accumulation[i] = summation

        if summation == 0:
            # print('Warning: zero credit')
            return self.environment.action_space.sample()

        # select action with probability proportional to hv
        random_number = self.generator.uniform(low=0, high=summation)

        # print('--AgentPQLEXP-_best_action----extra-accumulation-num-accion------')
        # print(extra)
        # print(accumulation)
        # print(num)

        for i in range(n):
            if random_number <= accumulation[i]:
                # self._acumula(extra[i][0], True)
                # print(extra[i][0])
                return info[i][0]

        print('Warning: agent_pql_exp._best_action: seleccionando acción de emergencia')
        return info[n - 1][0]

    def _non_greedy_action(self, state: object = None, extra: object = None) -> int:
        """
        Select action with probability inversely proportional to the credit indicated in extra.
        If necessary, probabilities are smoothed so that with epsilon = 0.5, a random walk is obtained.
        train_data is a  tuple. The first element is the maximum credit, the sencond a list of tuples: (action, credit),
        and the third the sum of credits.

        :param state:
        :param extra:
        :return:
        """

        # print('------------------------------------------------AgentPQLEXP3._greedy  (exploración)')

        n = self.environment.action_space.n

        e = extra[0]
        info = extra[1]
        s = extra[2]

        # cantidad a añadir para suavizar las probabilidades
        k = max(0, (e - ((2 * s) / n)))

        s2 = s + n * k  # suma de los creditos suavizados

        # print('n: {} s: {} e: {} k: {}'.format (n, s, e, k))

        # calculate array accumulated probabilites
        acu = np.zeros(n)
        # sum = 0  #suma de
        # for i in range(n):
        #    sum += extra[i][1]
        #    #sum += (maximum - extra[i][1])
        #    #acu[i] = sum

        if s == 0:
            # print('Warning: zero credit')
            return self.environment.action_space.sample()

        # calculate probabilities inversely proportional to score
        ns2 = s2 * n
        summation = 0
        # print('.................')
        for i in range(n):
            aux = (2 * s2 - n * (info[i][1] + k))
            summation += aux  # (s2 - nact * extra[i][1])
            acu[i] = summation
            # print('aux: {} acu[i] {}'.format(aux, acu[i]))

        # select action with probability proportional to hv
        num = self.generator.uniform(low=0, high=ns2)

        # print('extra: {}'.format(extra) )
        # print('s2: {} '.format(s2))
        # print('acu: {}'.format(acu))
        # print('ns2: {}'.format(ns2))

        for i in range(n):
            if num <= acu[i]:
                # print(extra[i][0])
                # self._acumula(extra[i][0], False)
                return info[i][0]

        print('Warning: agent_pql_exp2._non_greedy_action: seleccionando acción de emergencia')
        return info[n - 1][0]

    # def _acumula(self, accion, greedy):
    #     if self.contg == None:
    #         self.contg = [0,0,0,0]  #para problemas con 4 acciones
    #         self.conte = [0, 0, 0, 0]  # para problemas con 4 acciones
    #         self.cont2 = [0,0]
    #
    #     if greedy:
    #         self.cont2[1] += 1
    #         self.contg[accion] += 1
    #     else:
    #         self.cont2[0] += 1
    #         self.conte[accion] += 1
    #
    #
    #     print('Acciones g: {}, Acciones e: {}, explora-greedy: {}'.format(self.contg, self.conte, self.cont2))
