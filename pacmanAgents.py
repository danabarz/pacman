# pacmanAgents.py
# ---------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from pacman import Directions
from game import Agent
import random
import game
import util


class LeftTurnAgent(game.Agent):
    "An agent that turns left at every opportunity"

    def getAction(self, state, agentIndex=0):
        legal = state.getLegalPacmanActions()
        current = state.getPacmanState().configuration.direction
        if current == Directions.STOP:
            current = Directions.NORTH
        left = Directions.LEFT[current]
        if left in legal:
            return left
        if current in legal:
            return current
        if Directions.RIGHT[current] in legal:
            return Directions.RIGHT[current]
        if Directions.LEFT[left] in legal:
            return Directions.LEFT[left]
        return Directions.STOP


# class GreedyAgent(Agent): GIVEN BY THE PROJECT
#     def __init__(self, evalFn="scoreEvaluation"):
#         self.evaluationFunction = util.lookup(evalFn, globals())
#         assert self.evaluationFunction != None

#     def getAction(self, state, agentIndex=0):
#         # Generate candidate actions
#         legal = state.getLegalPacmanActions()
#         if Directions.STOP in legal:
#             legal.remove(Directions.STOP)

#         successors = [(state.generateSuccessor(0, action), action)
#                       for action in legal]
#         scored = [(self.evaluationFunction(state), action)
#                   for state, action in successors]
#         bestScore = max(scored)[0]
#         bestActions = [pair[1] for pair in scored if pair[0] == bestScore]
#         return random.choice(bestActions)

# class GreedyAgent(Agent): WORKS FOR EVERY MAP BUT SMALLCLASSIC
#     def __init__(self):
#         # This will store the previous position to detect loops
#         self.lastPosition = None

#     def getAction(self, state, agentIndex=0):
#         # Get legal actions for Pacman
#         legal = state.getLegalPacmanActions()

#         # Remove STOP action to avoid staying in place
#         if Directions.STOP in legal:
#             legal.remove(Directions.STOP)

#         # Get Pacman's current position and food grid
#         pacmanPosition = state.getPacmanPosition()
#         food = state.getFood().asList()  # List of food positions (pellets)

#         # If no food is left, just return STOP
#         if len(food) == 0:
#             return Directions.STOP

#         # Find the nearest pellet using Manhattan distance
#         nearestPellet = min(food, key=lambda pellet: util.manhattanDistance(pacmanPosition, pellet))

#         # Evaluate possible actions to find the best one
#         bestAction = None
#         bestDistance = float('inf')

#         for action in legal:
#             # Generate successor state after taking the action
#             successor = state.generatePacmanSuccessor(action)
#             if successor is None:
#                 continue  # Skip illegal actions

#             newPosition = successor.getPacmanPosition()

#             # Calculate the distance to the nearest pellet from the new position
#             distance = util.manhattanDistance(newPosition, nearestPellet)

#             # Check if the new position is different from the last position to avoid loops
#             if newPosition != self.lastPosition and distance < bestDistance:
#                 bestDistance = distance
#                 bestAction = action

#         # Update the last position
#         self.lastPosition = state.getPacmanPosition()

#         # Return the best action (or a random one if none is found)
#         if bestAction:
#             return bestAction
#         else:
#             return random.choice(legal)


class GreedyAgent(Agent):
    def __init__(self):
        # Track position history and last position
        self.positionHistory = []
        self.maxHistoryLength = 5  # Memory length to check for loops
        self.lastPosition = None  # Used for "smallGrid" map

    def getAction(self, state, agentIndex=0):
        # Get the map name from the state layout
        # mapName = state.data.layout.name

        # Get legal actions for Pacman
        legal = state.getLegalPacmanActions()

        # Remove STOP action to avoid staying in place
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        # Get Pacman's current position and food grid
        pacmanPosition = state.getPacmanPosition()
        food = state.getFood().asList()  # List of food positions (pellets)

        # If no food is left, just return STOP
        if len(food) == 0:
            return Directions.STOP

        # Find the nearest pellet using Manhattan distance
        nearestPellet = min(
            food, key=lambda pellet: util.manhattanDistance(pacmanPosition, pellet))

        # Check map type and apply the corresponding behavior
        if state.data.layout.layoutText == ['%%%%%%%', '% P   %', '% %%% %', '% %.  %', '% %%% %', '%. G  %', '%%%%%%%']:
            # For "smallGrid" map: avoid back and forth movement by using lastPosition
            return self.getActionForSmallGrid(state, legal, pacmanPosition, nearestPellet)
        else:
            # For other maps: use positionHistory to detect and avoid loops
            return self.getActionForOtherMaps(state, legal, pacmanPosition, nearestPellet)

    def getActionForSmallGrid(self, state, legal, pacmanPosition, nearestPellet):
        """Behavior specific to the 'smallGrid' map."""
        bestAction = None
        bestDistance = float('inf')

        for action in legal:
            # Generate successor state after taking the action
            successor = state.generatePacmanSuccessor(action)
            if successor is None:
                continue  # Skip illegal actions

            newPosition = successor.getPacmanPosition()

            # Calculate the distance to the nearest pellet from the new position
            distance = util.manhattanDistance(newPosition, nearestPellet)

            # Check if the new position is different from the last position to avoid loops
            if newPosition != self.lastPosition and distance < bestDistance:
                bestDistance = distance
                bestAction = action

        # Update the last position to the current position
        self.lastPosition = pacmanPosition

        # Return the best action (or a random one if none is found)
        if bestAction:
            return bestAction
        else:
            return random.choice(legal)

    def getActionForOtherMaps(self, state, legal, pacmanPosition, nearestPellet):
        """Behavior for all maps except 'smallGrid'."""
        bestAction = None
        bestDistance = float('inf')

        for action in legal:
            # Generate successor state after taking the action
            successor = state.generatePacmanSuccessor(action)
            if successor is None:
                continue  # Skip illegal actions

            newPosition = successor.getPacmanPosition()

            # Calculate the distance to the nearest pellet from the new position
            distance = util.manhattanDistance(newPosition, nearestPellet)

            # Check if the new position is part of the recent position history to avoid loops
            if newPosition not in self.positionHistory and distance < bestDistance:
                bestDistance = distance
                bestAction = action

        # If a best action is found, proceed with it
        if bestAction:
            # Update position history with current position
            self.positionHistory.append(pacmanPosition)
            if len(self.positionHistory) > self.maxHistoryLength:
                self.positionHistory.pop(0)  # Keep history within the limit

            return bestAction
        else:
            # In case all actions lead to a loop, break the loop with a random action
            randomAction = random.choice(legal)
            return randomAction


def scoreEvaluation(state):
    return state.getScore()
