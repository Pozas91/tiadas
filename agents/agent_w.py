"""
Agent W.

The W algorithm is an implementation of an adaptation of the approximation algorithm described by D.J White in
"Multi-objective infinite-horizon discounted Markov decision process".

We consider the case of a finite set of states `S`, a finite set of actions A(s) ∀s ∈ S, and a vector reward `r`
associated to each transition (s, a, s') from state `s` to state `s'` through action `a`.

The algorithm approximates V(s), the set of non-dominated vector values for every state `s`.

We consider the following operations and functions:
• ND(X), the set of non-dominated vectors from vector set X ⊂ R^n.
• r(s, a, s'), the vector reward associated to transition (s, a, s').
• p(s, a, s'), the transition probability associated to transition (s, a, s').
"""
import itertools

from agents import Agent
from environments import Environment
from models import Vector, GraphType
from functools import reduce


class AgentW(Agent):

    def __init__(self, environment: Environment, gamma: float = 1., seed: int = 0, initial_q_value: Vector = None):
        # Super call __init__
        super().__init__(environment=environment, gamma=gamma, seed=seed, initial_q_value=initial_q_value)

        # Check if initial_q_value is given
        self.initial_q_value = self.environment.default_reward.zero_vector if initial_q_value is None else initial_q_value

        # Vector with a vector set for each possible state `s`
        self.v = dict()

    def train(self, graph_type: GraphType, limit: int):

        for i in range(limit):
            v2 = self.v.copy()

            # Removes all items from the dictionary
            self.v.clear()

            # action_space has all
            for s in self.environment.states():

                # A(s) <- Extract all actions available from state `s`
                self.environment.current_state = s

                # Vector of Empty sets
                t = dict()

                # For each a in action_space
                for a in self.environment.action_space:

                    # Empty set for this a (T(a))
                    t_a = set()

                    # Get all reachable states for that pair of (s, a)
                    s2_set = self.environment.reachable_states(state=s, action=a)

                    lv = list()

                    for s2 in s2_set:
                        # If this state is unknown return empty set
                        lv.append(v2.get(s2, [self.environment.default_reward.zero_vector]))

                    # Calc cartesian product of each reachable states
                    cartesian_product = itertools.product(*lv)

                    for product in cartesian_product:

                        summation = self.environment.default_reward.zero_vector

                        for j, s2 in enumerate(s2_set):
                            # Probability to reach that state
                            p = self.environment.transition_probability(state=s, action=a, next_state=s2)

                            # Reward to reach that state
                            r = self.environment.transition_reward(state=s, action=a, next_state=s2)

                            # Get previous value per gamma
                            previous_value = product[j] * self.gamma

                            # Summation
                            summation += (r + previous_value) * p

                        # T(a) <- T(a) U {.....}
                        t_a = t_a.union({summation})

                        t.update({a: t_a})

                # V(s) <- ND[U T(a)]
                u_t = set.union(*t.values())
                self.v.update({s: Vector.m3_max(u_t)})

        return self.v
