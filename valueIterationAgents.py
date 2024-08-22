# valueIterationAgents.py
# -----------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import util
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
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # Old values Vk(s)
        for i in range(self.iterations):
            new_values = util.Counter()  # New values Vk+1(s)
            for state in self.mdp.getStates():
                if not self.mdp.isTerminal(state):
                    max_value = float('-inf')
                    for action in self.mdp.getPossibleActions(state):
                        q_value = 0
                        for next_state, prob in self.mdp.getTransitionStatesAndProbs(
                                state, action):
                            reward = self.mdp.getReward(state, action,
                                                        next_state)
                            q_value += prob * (
                                    reward + self.discount * self.values[
                                next_state])
                        max_value = max(max_value, q_value)
                    new_values[state] = max_value
            self.values = new_values

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]

    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the value function stored in self.values.
        """
        q_value = 0
        for next_state, prob in self.mdp.getTransitionStatesAndProbs(state,
                                                                     action):
            reward = self.mdp.getReward(state, action, next_state)
            q_value += prob * (
                    reward + self.discount * self.getValue(next_state))
        return q_value

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.
        """
        if self.mdp.isTerminal(state):
            return None
        best_action = None
        max_value = float('-inf')
        for action in self.mdp.getPossibleActions(state):
            q_value = self.computeQValueFromValues(state, action)
            if q_value > max_value:
                max_value = q_value
                best_action = action
        return best_action

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
