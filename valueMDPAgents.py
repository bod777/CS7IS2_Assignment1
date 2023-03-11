# valueMDPAgents.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

from pacman import Directions
from game_modules.game import Agent
import api
import time

class ValueMDPAgent(Agent):
    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self, fn='value'):
        # Maps:
        self.grid = set()                             # a set of all possible states in the game
        self.mappedStates = {}                        # a dict mapping all possible states to 4 directions
        self.reward = {}                              # a dict which stores reward of each cell
        self.utils = {}                               # a dict which stores utility of each cell
        self.actions = []
        self.cost = 0

        # Hyper-parameters:
        self.width = 0                                # width of the game map
        self.height = 0                               # height of the game map
        self.error = 0.001                            # error for convergence
        self.gamma = 0.9                              # discount factor
        self.actionProb = api.directionProb           # non-deterministic probability of policy: 0.8
        self.otherActionProb = (1-self.actionProb)/2  # 0.1

        # Parameters / Reward:
        self.food_reward = 0                          # Food reward
        self.empty_reward = 0                         # Empty cell reward

    # Gets run after an MDPAgent object is created and once there is
    # game state to access.
    def registerInitialState(self, state):
        print("Running Value Iteration MDPAgent!")
        startTime = time.time()
        # 1. Create a set of all states of the game                         -> self.grid
        self.makeGrid(state)
        # 2. Initialize rewards values based on the size of the map         -> self.xxx_reward and hyper-parameters
        self.initializeReward(state)
        # 3. Initialize rewards of each state (except for walls)            -> self.reward (dict)
        self.updateReward(state)
        # 4. Initialize all grids' utility to 0                             -> self.utility (dict)
        self.resetUtility(state)
        # 5. Map Pacman's all possible states to its 4 adjacent directions  -> self.mappedStates
        self.mapState(state)
        self.valueIteration(state)
        self.timeTaken = time.time() - startTime

    # This is what gets run in between multiple games
    def final(self, state):
        print("Maze Solved.")
        print('Cost: %.0f' % (self.cost))
        print('Time to find optimal path: %.5f seconds' % self.timeTaken)
        print('Nodes Visited: %.0f' % (len(self.mappedStates)))

    def getAction(self, state):
        policy = self.choosePolicy(state)
        legal = api.legalActions(state)
        self.cost += 1
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        return api.makeMove(policy, legal)

    def valueIteration(self, state):
        states = self.mappedStates  
        reward = self.reward 
        utils = self.utils
        prevUtils = dict(utils)

        while True:
            delta = 0
            for coord, utility in prevUtils.items():
                tempUtils = []
                currentReward = reward[coord]
                for _, potentialGrids in states[coord].items():
                    tempUtils.append(currentReward + self.gamma * (
                            self.actionProb * prevUtils[potentialGrids[0]] + self.otherActionProb *
                            prevUtils[potentialGrids[1]] + self.otherActionProb * prevUtils[potentialGrids[2]]))
                utils[coord] = max(tempUtils)
                delta = max(delta, abs(utils[coord] - utility))
            prevUtils = dict(utils)
            if delta < self.error:
                self.utils = dict(utils)
                break

    def choosePolicy(self, state):
        # choose a direction which returns a maximum expected utility
        pacman = api.whereAmI(state)
        (x, y) = pacman
        walls = api.walls(state)
        north = (x, y+1)
        south = (x, y-1)
        east = (x+1, y)
        west = (x-1, y)

        # check if the next location is a wall, if its a wall, stop; otherwise, go
        if north in walls:
            north = pacman
        if south in walls:
            south = pacman
        if east in walls:
            east = pacman
        if west in walls:
            west = pacman

        # calculate expected utility without reward, since reward is the same
        North_EU = self.actionProb * self.utils[north] + \
                   self.otherActionProb * self.utils[west] + \
                   self.otherActionProb * self.utils[east]
        South_EU = self.actionProb * self.utils[south] + \
                   self.otherActionProb * self.utils[west] + \
                   self.otherActionProb * self.utils[east]
        East_EU = self.actionProb * self.utils[east] + \
                  self.otherActionProb * self.utils[north] + \
                  self.otherActionProb * self.utils[south]
        West_EU = self.actionProb * self.utils[west] + \
                  self.otherActionProb * self.utils[north] + \
                  self.otherActionProb * self.utils[south]

        # get the index of max expected utility
        list = [North_EU, South_EU, East_EU, West_EU]
        maxIndex = 0
        for i in range(len(list)):
            if list[i] > list[maxIndex]:
                maxIndex = i

        # return direction: 0-north  1-south  2-east  3-west
        if maxIndex == 0:
            self.actions.append(Directions.NORTH)
            return Directions.NORTH
        if maxIndex == 1:
            self.actions.append(Directions.SOUTH)
            return Directions.SOUTH
        if maxIndex == 2:
            self.actions.append(Directions.EAST)
            return Directions.EAST
        if maxIndex == 3:
            self.actions.append(Directions.WEST)
            return Directions.WEST

    def initializeReward(self, state):
        self.food_reward = 5
        self.empty_reward = -2

    def updateReward(self, state):
        """
            Update reward for every cell that is not a wall.
            Food reward: food reward will be set differently based on the number of surrounded walls, e.g. food in the
                corners will assigned lower value than those at a crossing. Additionally, we will boost the reward of
                the last piece of food since it's the terminal state of the game.
        """

        walls = api.walls(state)
        foods = api.food(state)

        # Empty reward
        self.reward = {state: -1 for state in self.grid if state not in walls}
        # Food reward
        for food in foods:
            count = self.countWalls(state, food)
            if count <= 2:
                # If the food is only surrounded by 2 or less than 2 walls, just set the reward for this food
                self.reward[food] = self.food_reward
            else:
                # If the food is only surrounded by greater than 2 walls, suppress the reward for this food
                self.reward[food] = self.food_reward / (1 + count * count)
        if self.checkLastFood(state):
            self.reward[foods[0]] += self.food_reward

    def checkLastFood(self, state):
        """ check if there is only one piece of food """

        foods = api.food(state)
        if len(foods) == 1:
            return True
        else:
            return False

    def countWalls(self, state, food):
        """ Count the number of walls that near a piece of food """

        walls = api.walls(state)
        north = (food[0], food[1] + 1)
        south = (food[0], food[1] - 1)
        west = (food[0] + 1, food[1])
        east = (food[0] - 1, food[1])
        count = 0
        if north in walls:
            count += 1
        if south in walls:
            count += 1
        if west in walls:
            count += 1
        if east in walls:
            count += 1
        return count
    
    def resetUtility(self, state):
        """ Initialize all cells' utility to 0 """

        walls = api.walls(state)
        self.utils = {state: 0 for state in self.grid if state not in walls}

    def mapState(self, state):
        """ # Map reachable states for 4 directions of all states """

        walls = api.walls(state)
        states = dict.fromkeys(self.reward.keys())

        # Iterate all cells in the map and map their potential states in 4 directions.
        for cell in states.keys():
            neighbours = self.fourNeighbours(cell)
            states[cell] = {'north': [neighbours[3], neighbours[0], neighbours[2]],
                            'south': [neighbours[1], neighbours[0], neighbours[2]],
                            'east': [neighbours[0], neighbours[3], neighbours[1]],
                            'west': [neighbours[2], neighbours[3], neighbours[1]]}

            # Iterate all 4 directions and iterate all 3 potential states in one direction if any potential state
            # is wall, set it the original cell
            for _, possibleStates in states[cell].items():
                for state in possibleStates:
                    if state in walls:
                        possibleStates[possibleStates.index(state)] = cell
        self.mappedStates = states

    def fourNeighbours(self, cell):
        """ Get 4 adjacent neighbours of a cell """

        (x, y) = cell
        east = (x + 1, y)
        south = (x, y - 1)
        west = (x - 1, y)
        north = (x, y + 1)
        return [east, south, west, north]

    def makeGrid(self, state):
        """
            Create a grid of the whole game map.
            Notice the data structure for grid is set because it can be quickly accessed.
        """

        corners = api.corners(state)
        BL = corners[0]         # Bottom left corner
        TR = corners[3]         # Top right corner
        self.width = TR[0]+1    # store the width of the game map
        self.height = TR[1]+1   # store the height of the game map
        width = range(BL[0], TR[0]+1)
        height = range(BL[1], TR[1]+1)
        self.grid = set((x, y) for x in width for y in height)  # build up the game grid in a python set



