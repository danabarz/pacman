# ghostAgents.py
# --------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# For more info, see http://inst.eecs.berkeley.edu/~cs188/sp09/pacman.html

from game import Agent
from game import Actions
from game import Directions
import random
from pacman import GhostRules
from util import manhattanDistance
import util
import search

class GhostAgent( Agent ):
    def __init__( self, index ):
        self.index = index

    def getAction( self, state ):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution( dist )

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()

class RandomGhost( GhostAgent ):
    "A ghost that chooses a legal action uniformly at random."
    def getDistribution( self, state ):
        dist = util.Counter()
        for a in state.getLegalActions( self.index ): dist[a] = 1.0
        dist.normalize()
        return dist

class DirectionalGhost( GhostAgent ):
    "A ghost that prefers to rush Pacman, or flee when scared."
    def __init__( self, index, prob_attack=0.8, prob_scaredFlee=0.8 ):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution( self, state ):
        # Read variables from state
        ghostState = state.getGhostState( self.index )
        legalActions = state.getLegalActions( self.index )
        pos = state.getGhostPosition( self.index )
        isScared = ghostState.scaredTimer > 0

        speed = GhostRules.GHOST_SPEED
        if isScared: speed /= 2.0

        actionVectors = [Actions.directionToVector( a, speed ) for a in legalActions]
        newPositions = [( pos[0]+a[0], pos[1]+a[1] ) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance( pos, pacmanPosition ) for pos in newPositions]
        if isScared:
            bestScore = max( distancesToPacman )
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min( distancesToPacman )
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip( legalActions, distancesToPacman ) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions: dist[a] = bestProb / len(bestActions)
        for a in legalActions: dist[a] += ( 1-bestProb ) / len(legalActions)
        dist.normalize()
        return dist

class AStarGhost(GhostAgent):
    def __init__(self, index):
        super().__init__(index)

    def ghostHeuristic(self, ghostPosition, pacmanPosition, coins):
        """
        Heuristic that estimates the cost from the ghost's current position to the nearest coin relative to Pacman.
        """
        if not coins:
            return 0

        # Find the coin closest to Pacman
        closest_coin_to_pacman = min(coins, key=lambda coin: manhattanDistance(pacmanPosition, coin))

        # Compute the distance from the ghost to the closest coin to Pacman
        ghost_to_coin_distance = manhattanDistance(ghostPosition, closest_coin_to_pacman)

        return ghost_to_coin_distance

    def aStarSearch(self, startState, goalState, state):
        """
        A* search to find the optimal path to the goal from the ghost's current position.
        """
        open_list = util.PriorityQueue()
        open_list.push((startState, []), 0)
        visited = set()

        while not open_list.isEmpty():
            currentState, actions = open_list.pop()

            if currentState in visited:
                continue

            visited.add(currentState)

            if currentState == goalState:
                return actions

            for action in state.getLegalActions(self.index):
                if action == Directions.STOP:  # Skip the "stop" action
                    continue

                dx, dy = Actions.directionToVector(action)
                next_x, next_y = int(currentState[0] + dx), int(currentState[1] + dy)
                nextState = (next_x, next_y)

                if nextState not in visited and not state.hasWall(next_x, next_y):
                    newActions = actions + [action]
                    g_cost = len(newActions)
                    h_cost = self.ghostHeuristic(nextState, state.getPacmanPosition(), state.getFood().asList())
                    f_cost = g_cost + h_cost
                    open_list.push((nextState, newActions), f_cost)

        return []

    def getDistribution(self, state):
        """
        Determine the distribution of actions for the ghost using A* search.
        """
        ghostPosition = state.getGhostPosition(self.index)
        pacmanPosition = state.getPacmanPosition()
        coins = state.getFood().asList()
        ghostState = state.getGhostState(self.index)
        isScared = ghostState.scaredTimer > 0

        # Adjust speed when scared
        speed = 0.5 if isScared else 1

        if isScared:
            # Flee from Pacman when scared
            actions = state.getLegalActions(self.index)
            actions = [a for a in actions if a != Directions.STOP]
            if not actions:
                return util.Counter()

            actionVectors = [Actions.directionToVector(a, speed) for a in actions]
            newPositions = [(ghostPosition[0] + a[0], ghostPosition[1] + a[1]) for a in actionVectors]
            distancesToPacman = [manhattanDistance(pos, pacmanPosition) for pos in newPositions]

            # Select action that maximizes distance from Pacman
            bestDistance = max(distancesToPacman)
            bestActions = [a for a, dist in zip(actions, distancesToPacman) if dist == bestDistance]

            dist = util.Counter()
            for a in bestActions:
                dist[a] = 1.0 / len(bestActions)
            return dist
        else:
            # If not scared, chase the closest coin to Pacman
            if not coins:
                return util.Counter()

            closest_coin_to_pacman = min(coins, key=lambda coin: manhattanDistance(pacmanPosition, coin))

            # Use A* search to find the optimal path to this coin
            actions = self.aStarSearch(ghostPosition, closest_coin_to_pacman, state)

            dist = util.Counter()
            if len(actions) > 0:
                dist[actions[0]] = 1.0
            else:
                legalActions = [a for a in state.getLegalActions(self.index) if a != Directions.STOP]
                if legalActions:
                    dist[random.choice(legalActions)] = 1.0

            return dist