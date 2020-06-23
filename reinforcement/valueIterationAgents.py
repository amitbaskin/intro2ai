# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import mdp, util

from learningAgents import ValueEstimationAgent


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
        """
        super().__init__()
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.epsilon = 0.01
        self.values, self.policy = self.value_iteration()

    def value_iteration(self):
        states = self.mdp.getStates()
        U_prime = util.Counter()
        stop = True
        iteration = 0

        while iteration < self.iterations:
            U = U_prime.copy()
            iteration += 1

            for state in states:
                sums = util.Counter()
                for action in self.mdp.getPossibleActions(state):
                    if state == "TERMINAL_STATE":
                        continue
                    next_state_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
                    for state_inner, prob in next_state_and_probs:
                        sums[action] += prob * U[state_inner]
                if len(sums) == 0:
                    max_value = 0
                else:
                    max_value = max(sums.values())
                U_prime[state] = self.mdp.getReward(state, None, None) + self.discount * max_value

            for state in states:
                if abs(U_prime[state] - U[state]) >= self.epsilon:
                    stop = False
                    break

            if stop:
                break
            else:
                stop = True

        policy = dict()
        for state in states:
            sums = util.Counter()
            for action in self.mdp.getPossibleActions(state):
                if state == "TERMINAL_STATE":
                    continue
                next_state_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
                for state_inner, prob in next_state_and_probs:
                    sums[action] += prob * U_prime[state_inner]
            max_action = max(sums, key=sums.get) if len(sums) != 0 else None
            policy[state] = max_action

        return U_prime, policy

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def getQValue(self, state, action):
        """
          The q-value of the state action pair
          (after the indicated number of value iteration
          passes).  Note that value iteration does not
          necessarily create this quantity and you may have
          to derive it on the fly.
        """
        # sumP = 0
        # for state_inner in self.mdp.getStates():
        #     if state_inner == "TERMINAL_STATE":
        #         continue
        #     a = util.Counter(dict(self.mdp.getTransitionStatesAndProbs(state, action)))
        #     sumP += a[state_inner] * self.values[state_inner]
        # return self.mdp.getReward(state, action, None) + self.discount *
        next_state_and_probs = self.mdp.getTransitionStatesAndProbs(state, action)
        sumP = 0
        for _, prob in next_state_and_probs:
            sumP += prob
        return self.mdp.getReward(state, None, None) + self.discount * sumP

    def getPolicy(self, state):
        """
          The policy is the best action in the given state
          according to the values computed by value iteration.
          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        return self.policy[state]

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.getPolicy(state)
