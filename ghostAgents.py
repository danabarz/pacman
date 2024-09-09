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
from util import manhattanDistance
import util


class GhostAgent(Agent):
    def __init__(self, index):
        self.index = index

    def getAction(self, state, agentIndex=1):
        dist = self.getDistribution(state)
        if len(dist) == 0:
            return Directions.STOP
        else:
            return util.chooseFromDistribution(dist)

    def getDistribution(self, state):
        "Returns a Counter encoding a distribution over actions from the provided state."
        util.raiseNotDefined()


class RandomGhost(GhostAgent):
    "A ghost that chooses a legal action uniformly at random."

    def getDistribution(self, state):
        dist = util.Counter()
        for a in state.getLegalActions(self.index):
            dist[a] = 1.0
        dist.normalize()
        return dist


class DirectionalGhost(GhostAgent):
    "A ghost that prefers to rush Pacman, or flee when scared."

    def __init__(self, index, prob_attack=0.8, prob_scaredFlee=0.8):
        self.index = index
        self.prob_attack = prob_attack
        self.prob_scaredFlee = prob_scaredFlee

    def getDistribution(self, state):
        # Read variables from state
        ghostState = state.getGhostState(self.index)
        legalActions = state.getLegalActions(self.index)
        pos = state.getGhostPosition(self.index)
        isScared = ghostState.scaredTimer > 0

        speed = 0.5 if isScared else 1

        actionVectors = [Actions.directionToVector(
            a, speed) for a in legalActions]
        newPositions = [(pos[0]+a[0], pos[1]+a[1]) for a in actionVectors]
        pacmanPosition = state.getPacmanPosition()

        # Select best actions given the state
        distancesToPacman = [manhattanDistance(
            pos, pacmanPosition) for pos in newPositions]
        if isScared:
            bestScore = max(distancesToPacman)
            bestProb = self.prob_scaredFlee
        else:
            bestScore = min(distancesToPacman)
            bestProb = self.prob_attack
        bestActions = [action for action, distance in zip(
            legalActions, distancesToPacman) if distance == bestScore]

        # Construct distribution
        dist = util.Counter()
        for a in bestActions:
            dist[a] = bestProb / len(bestActions)
        for a in legalActions:
            dist[a] += (1-bestProb) / len(legalActions)
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
        closest_coin_to_pacman = min(
            coins, key=lambda coin: manhattanDistance(pacmanPosition, coin))

        # Compute the distance from the ghost to the closest coin to Pacman
        ghost_to_coin_distance = manhattanDistance(
            ghostPosition, closest_coin_to_pacman)

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
                next_x, next_y = int(
                    currentState[0] + dx), int(currentState[1] + dy)
                nextState = (next_x, next_y)

                if nextState not in visited and not state.hasWall(next_x, next_y):
                    newActions = actions + [action]
                    g_cost = len(newActions)
                    h_cost = self.ghostHeuristic(
                        nextState, state.getPacmanPosition(), state.getFood().asList())
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

            actionVectors = [Actions.directionToVector(
                a, speed) for a in actions]
            newPositions = [(ghostPosition[0] + a[0], ghostPosition[1] + a[1])
                            for a in actionVectors]
            distancesToPacman = [manhattanDistance(
                pos, pacmanPosition) for pos in newPositions]

            # Select action that maximizes distance from Pacman
            bestDistance = max(distancesToPacman)
            bestActions = [a for a, dist in zip(
                actions, distancesToPacman) if dist == bestDistance]

            dist = util.Counter()
            for a in bestActions:
                dist[a] = 1.0 / len(bestActions)
            return dist
        else:
            # If not scared, chase the closest coin to Pacman
            if not coins:
                return util.Counter()

            closest_coin_to_pacman = min(
                coins, key=lambda coin: manhattanDistance(pacmanPosition, coin))

            # Use A* search to find the optimal path to this coin
            actions = self.aStarSearch(
                ghostPosition, closest_coin_to_pacman, state)

            dist = util.Counter()
            if len(actions) > 0:
                dist[actions[0]] = 1.0
            else:
                legalActions = [a for a in state.getLegalActions(
                    self.index) if a != Directions.STOP]
                if legalActions:
                    dist[random.choice(legalActions)] = 1.0

            return dist


class MinMaxGhost(GhostAgent):
    def __init__(self, index, depth=2):
        super().__init__(index)
        self.depth = depth

    def evaluationFunction(self, state):
        """
        Evaluation function for the MinMaxGhost.
        When the ghost is not scared, it tries to maximize the number of food pellets by moving towards Pacman.
        When scared, it tries to flee from Pacman.
        """
        ghostState = state.getGhostState(self.index)
        ghostPosition = state.getGhostPosition(self.index)
        pacmanPosition = state.getPacmanPosition()
        numFood = state.getNumFood()

        # If ghost is scared, prioritize fleeing from Pacman
        if ghostState.scaredTimer > 0:
            return manhattanDistance(ghostPosition, pacmanPosition)

        # Otherwise, prioritize moving towards Pacman while considering food count
        return -manhattanDistance(ghostPosition, pacmanPosition) + numFood * 1000

    def minmax(self, state, depth, agentIndex):
        """
        Minimax algorithm implementation where the ghost minimizes/maximizes based on the evaluation function.
        """
        if state.isWin() or state.isLose() or depth == 0:
            return self.evaluationFunction(state)

        if agentIndex == 0:  # Pacman's turn (Maximizer)
            return self.maxValue(state, depth)
        else:  # Ghost's turn (Minimizer)
            return self.minValue(state, depth, agentIndex)

    def maxValue(self, state, depth):
        """
        Max function for Pacman.
        Pacman tries to minimize the number of food pellets left on the map.
        """
        v = float('-inf')
        legalActions = state.getLegalActions(0)  # Pacman's legal actions

        for action in legalActions:
            successorState = state.generateSuccessor(0, action)
            # Proceed to Ghost's turn
            v = max(v, self.minmax(successorState, depth, 1))

        return v

    def minValue(self, state, depth, agentIndex):
        """
        Min function for the ghost.
        The ghost tries to maximize the number of food pellets left on the map.
        When scared, the ghost tries to flee from Pacman.
        """
        v = float('inf')
        legalActions = state.getLegalActions(agentIndex)

        nextAgent = (agentIndex + 1) % state.getNumAgents()
        # Reduce depth only when all agents have moved
        nextDepth = depth - 1 if nextAgent == 0 else depth

        for action in legalActions:
            successorState = state.generateSuccessor(agentIndex, action)
            v = min(v, self.minmax(successorState, nextDepth, nextAgent))

        return v

    def getAction(self, state, agentIndex=1):
        """
        Returns the minimax action from the current gameState using self.depth and self.evaluationFunction.
        """
        legalActions = state.getLegalActions(self.index)
        bestAction = None
        bestValue = float('inf')

        for action in legalActions:
            successorState = state.generateSuccessor(self.index, action)
            value = self.minmax(successorState, self.depth,
                                (self.index + 1) % state.getNumAgents())

            if value < bestValue:
                bestValue = value
                bestAction = action

        return bestAction
