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

    def __init__(self, q_table_name=None, **args):
        "You can initialize Q-values here..."
        ReinforcementAgent.__init__(self, **args)
        self.qvalues = {}
        self.q_table_name = q_table_name
        if q_table_name:
            self.load_q_table(q_table_name)

    def save_q_table(self, file_name):
        """Saves the Q-table to a file with backup and error handling."""
        try:
            # Save to a temporary file first
            temp_file_name = file_name + ".tmp"
            with open(temp_file_name, 'wb') as f:
                print("Saving Q-table to temporary file:", temp_file_name)
                pickle.dump(self.qvalues, f)

            # Replace the old file with the new one
            import os
            os.replace(temp_file_name, file_name)
            print(f"Q-table successfully saved to {file_name}")

        except IOError as e:
            print(f"Error saving Q-table to {file_name}: {e}")

    def load_q_table(self, file_name):
        """Loads the Q-table from a file with enhanced error handling."""
        try:
            with open(file_name, 'rb') as f:
                self.qvalues = pickle.load(f)
                # print("Type of data:", type(self.qvalues))
                # if isinstance(self.qvalues , dict):
                #     print("Contents of the dictionary:")
                #     for key, value in self.qvalues .items():
                #         print(f"Key: {key}, Value: {value}")
                # elif isinstance(self.qvalues , list):
                #     print("Contents of the list:")
                #     for item in self.qvalues :
                #         print(item)
                # else:
                #     print("Data:", self.qvalues )
            print(f"Q-table loaded from {file_name}")
        except FileNotFoundError:
            print(
                f"No Q-table found at {file_name}. Starting with an empty Q-table.")
            self.qvalues = {}
        except (pickle.UnpicklingError, IOError) as e:
            print(
                f"Error loading Q-table from {file_name}. Starting with an empty Q-table. Error: {e}")
            self.qvalues = {}

    def getQValue(self, state, action):
        """
        Returns Q(state,action)
        Should return 0.0 if we have never seen a state
        or the Q node value otherwise
        """
        return self.qvalues.get((state, action), 0.0)

    def setQValue(self, state, action, value):
        self.qvalues[(state, action)] = value

    def computeValueFromQValues(self, state):
        """
        Returns max_action Q(state,action)
        where the max is over legal actions.  Note that if
        there are no legal actions, which is the case at the
        terminal state, you should return a value of 0.0.
        """
        qvalues = [self.getQValue(state, action)
                   for action in self.getLegalActions(state)]
        if not len(qvalues):
            return 0.0
        return max(qvalues)

    def computeActionFromQValues(self, state):
        """
        Compute the best action to take in a state. If there are no legal actions,
        return None.
        """
        best_value = self.getValue(state)
        legal_actions = self.getLegalActions(state)

        if not legal_actions:
            return None

        best_actions = [action for action in legal_actions if self.getQValue(
            state, action) == best_value]
        return random.choice(best_actions) if best_actions else None

    def getAction(self, state, agentIndex=1):
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

    def update(self, state, action, nextState, reward, done=False):
        """
        Update the Q-value for the state-action pair using the Bellman equation:
        Q(s, a) = (1 - alpha) * Q(s, a) + alpha * (reward + discount * max_{a'} Q(s', a'))
        where s' is the next state, a' is the next action, and alpha is the learning rate.
        """
        disc = self.discount
        alpha = self.alpha

        qvalue = self.getQValue(state, action)
        next_value = self.getValue(nextState)
        td_error = reward + disc * next_value - qvalue
        new_value = qvalue + alpha * td_error
        self.setQValue(state, action, new_value)
        return reward

    def getPolicy(self, state):
        # Get best action from state
        return self.computeActionFromQValues(state)

    def getValue(self, state):
        # Get best q-value from state
        return self.computeValueFromQValues(state)


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

    def getAction(self, state, agentIndex=0):
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

    def update(self, state, action, nextState, reward, done=False):
        """
        Should update your weights based on transition
        """
        features = self.featExtractor.getFeatures(state, action)
        correction = reward + self.discount * \
            self.getValue(nextState) - self.getQValue(state, action)
        for feature in features:
            self.weights[feature] += self.alpha * \
                correction * features[feature]
        return reward

    def final(self, state):
        "Called at the end of each game."
        # call the super-class final method
        PacmanQAgent.final(self, state)

        # did we finish training?
        if self.episodesSoFar == self.numTraining:
            # you might want to print your weights here for debugging
            print(self.weights)


class GhostQAgent(QLearningAgent):
    "Exactly the same as QLearningAgent, but with different default parameters"

    def __init__(self, index, epsilon=0.1, gamma=0.9, alpha=0.1, numTraining=0, **args):
        """
        These default parameters can be changed from the pacman.py command line.
        """
        args['epsilon'] = epsilon
        args['gamma'] = gamma
        args['alpha'] = alpha
        args['numTraining'] = numTraining
        self.index = index
        QLearningAgent.__init__(
            self, actionFn=lambda state: state.getLegalActions(index), **args)

    def getAction(self, state, agentIndex=1):
        """
        Calls the getAction method of QLearningAgent and then
        informs parent of action for Pacman.
        """
        action = QLearningAgent.getAction(self, state)
        self.doAction(state, action)
        return action

    def update(self, state, action, nextState, reward, done=False):
        ghost_pos = nextState.getGhostPosition(self.index)
        old_distance = util.manhattanDistance(
            state.getPacmanPosition(), ghost_pos)
        new_distance = util.manhattanDistance(
            nextState.getPacmanPosition(), ghost_pos)
        reward = reward + new_distance if new_distance < old_distance else reward - new_distance
        return QLearningAgent.update(self, state, action, nextState, reward)

    def final(self, state):
        """
        This method is called by the game after a learning episode ends.
        It saves the Q-table every 100 episodes and prints the average score.
        """
        # Call the parent class's final method
        QLearningAgent.final(self, state)

        # Save the Q-table
        if self.episodesSoFar % 100 == 0 and self.q_table_name:
            print("saving Q-table to file: {}".format(self.q_table_name))
            self.save_q_table(self.q_table_name)
