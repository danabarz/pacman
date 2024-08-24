# qlearningAgents.py
# ------------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

import pickle
import random

import util
from learningAgents import ReinforcementAgent


class QLearningAgent(ReinforcementAgent):
    """
      Q-Learning Agent

      Functions you should fill in:
        - computeValueFromQValues
        - computeActionFromQValues
        - getQValue
        - getAction
        - update

      Instance variables you have access to
        - self.epsilon (exploration prob)
        - self.alpha (learning rate)
        - self.discount (discount rate)

      Functions you should use
        - self.getLegalActions(state)
          which returns legal actions for a state
    """

    def __init__(self, q_table_file=None, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qvalues = {}
        self.q_table = {}
        if q_table_file:
            self.load_q_table(q_table_file)

    def save_q_table(self, file_name):
        """Saves the Q-table to a file."""
        with open(file_name, 'wb') as f:
            pickle.dump(self.q_table, f)
        print(f"Q-table saved to {file_name}")

    def load_q_table(self, file_name):
        """Loads the Q-table from a file."""
        try:
            with open(file_name, 'rb') as f:
                self.q_table = pickle.load(f)
            print(f"Q-table loaded from {file_name}")
        except FileNotFoundError:
            print(f"No Q-table found at {file_name}. Starting with an empty Q-table.")
            self.q_table = {}

    def getQValue(self, state, action):
        """
          Returns Q(state,action)
          Should return 0.0 if we have never seen a state
          or the Q node value otherwise
        """
        if (state, action) in self.qvalues:
            return self.qvalues[(state, action)]
        else:
            return 0.0

    def setQValue(self, state, action, value):
        self.qvalues[(state, action)] = value

    def computeValueFromQValues(self, state):
        """
          Returns max_action Q(state,action)
          where the max is over legal actions.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return a value of 0.0.
        """
        qvalues = [self.getQValue(state, action) for action in self.getLegalActions(state)]
        if not len(qvalues): return 0.0
        return max(qvalues)

    def computeActionFromQValues(self, state):
        """
          Compute the best action to take in a state.  Note that if there
          are no legal actions, which is the case at the terminal state,
          you should return None.
        """
        best_value = self.getValue(state)
        best_actions = [action for action in self.getLegalActions(state) \
                        if self.getQValue(state, action) == best_value]

        if not len(best_actions):
            return None
        else:
            return random.choice(best_actions)

    def getAction(self, state):
        """
          Compute the action to take in the current state.  With
          probability self.epsilon, we should take a random action and
          take the best policy action otherwise.  Note that if there are
          no legal actions, which is the case at the terminal state, you
          should choose None as the action.

          HINT: You might want to use util.flipCoin(prob)
          HINT: To pick randomly from a list, use random.choice(list)
        """
        # Pick Action
        legal_actions = self.getLegalActions(state)
        action = None

        if util.flipCoin(self.epsilon):
            action = random.choice(legal_actions)
        else:
            action = self.getPolicy(state)

        return action

    def update(self, state, action, nextState, reward):
        """
          The parent class calls this to observe a
          state = action => nextState and reward transition.
          You should do your Q-Value update here

          NOTE: You should never call this function,
          it will be called on your behalf
          
          next_value = max[a'] Q(s', a')
          donde s' es el siguiente estado
          
          The update se realiza al llegar al estado s' y es realizado por la ecuacion:
            
            - Q(s, a) = (1-alpha) * Q(s, a) + alpha * (R(s,a,s') + disc * max{a'}[Q(s',a')])
        """
        disc = self.discount
        alpha = self.alpha
        qvalue = self.getQValue(state, action)
        next_value = self.getValue(nextState)

        # new_value = qvalue + alpha * (reward + disc * next_value - qvalue)
        new_value = (1 - alpha) * qvalue + alpha * (reward + disc * next_value)

        self.setQValue(state, action, new_value)

    def getPolicy(self, state):
        return self.computeActionFromQValues(state)  # Get best action from state

    def getValue(self, state):
        return self.computeValueFromQValues(state)  # Get best q-value from state


class PacmanQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, epsilon=0.1, gamma=0.9, alpha=0.1, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        For example, to change the exploration rate, try:
            python pacman.py -p PacmanQLearningAgent -a epsilon=0.1

        epsilon  - exploration rate
        gamma    - discount factor
        alpha    - learning rate
        numTraining - number of training episodes, i.e. no learning after these many episodes
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = 0  # This is always Pacman
        QLearningAgent.__init__(self, **args)

    def getAction(self, state):
        """
        Simply calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.  Do not change or remove this
        method.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action


class ApproximateQAgent(PacmanQAgent):
    """
       ApproximateQLearningAgent

       You should only have to overwrite getQValue
       and update.  All other QLearningAgent functions
       should work as is.
    """

    def __init__(self, extractor='IdentityExtractor', **args):
        self.featExtractor = util.lookup(extractor, globals())()
        PacmanQAgent.__init__(self, **args)
        self.setWeights()

    def setWeights(self, weights={}):
        self.weights = util.Counter(weights)

    def getWeights(self):
        return self.weights

    def getQValue(self, state, action):
        """
        Should return Q(state,action) = w * featureVector
        where * is the dotProduct operator
        """
        features = self.featExtractor.getFeatures(state, action)
        result = 0
        for feature in features:
            result += self.weights[feature] * features[feature]
        return result

    def update(self, state, action, nextState, reward):
        """
        Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        correction = reward + self.discount * self.getValue(nextState) - self.getQValue(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * correction * features[feature]

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            pass


class GhostQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"
    EP_COUNTER = 0

    def __init__(self, index, epsilon=0.25, gamma=0.8, alpha=0.25, numTraining=0, q_table_file=None, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = index
        self.q_table_file = q_table_file
        QLearningAgent.__init__(self, actionFn=lambda state: state.getLegalActions(index), **args)

        # Load Q-table if a file is provided
        self.q_table = {}
        self.episodeScores = []
        if q_table_file:
            self.load_q_table(q_table_file)


    def getAction(self, state):
        """
        Calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action

    def final(self, state):
        """
        This method is called by the game after a learning episode ends.
        It saves the Q-table every 100 episodes and prints the average score.
        """
        # Record the score for the current episode
        self.episodeScores.append(state.getScore())

        # Every 100 episodes, calculate the average score and save the Q-table
        if self.episodesSoFar % 100 == 0:
            average_score = sum(self.episodeScores) / len(self.episodeScores)
            print(f"Episode {self.episodesSoFar} finished.")
            print(f"  - Average score over the last 100 episodes: {average_score}")
            print(f"  - Number of training episodes left: {max(0, self.numTraining - self.episodesSoFar)}")

            # Save the Q-table
            if self.q_table_file:
                self.save_q_table(self.q_table_file)

            # Reset the scores for the next batch of 100 episodes
            self.episodeScores = []

        # Call the parent class's final method
        QLearningAgent.final(self, state)
