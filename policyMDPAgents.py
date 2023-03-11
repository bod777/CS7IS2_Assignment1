# policyMDPAgent.py
# ---------

from pacman import Directions
from game_modules.game import Agent
import api
import numpy as np
import time

class PolicyMDPAgent(Agent):
    def __init__(self):
        # Maps:
        self.grid = set()                             # a set of all possible states in the game
        self.mappedStates = {}                        # a dict mapping all possible states to 4 directions
        self.reward = {}                              # a dict which stores reward of each cell
        self.utils = {}                               # a dict which stores utility of each cell
        self.possibleActions = {'north', 'south', 'east', 'west'}
        self.policy = {}
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
        print("Running Policy Iteration MDPAgent!")
        self.startState = state.getPacmanPosition()
        startTime = time.time()
        # Create a set of all states of the game                         -> self.grid
        self.makeGrid(state)
        # Initialize rewards values based on the size of the map         -> self.xxx_reward and hyper-parameters
        self.initializeReward(state)
        # Initialize rewards of each state (except for walls)            -> self.reward (dict)
        self.updateReward(state)
        # Initialize all grids' utility to 0                             -> self.utility (dict)
        self.resetUtility(state)
        # Map Pacman's all possible states to its 4 adjacent directions  -> self.mappedStates
        self.mapState(state)
        # Create a basic policy of always go west
        self.initializePolicy(state)  
        self.policyIteration(state)
        self.timeTaken = (time.time() - startTime)

    def final(self, state):
        print("Maze Solved.")
        print('Cost: %.0f' % (self.cost))
        print('Time to find optimal path: %.5f seconds' % self.timeTaken)
        print('Nodes Visited: %.0f' % (len(self.mappedStates)))
    
    def getAction(self, state):
        policy = self.getPolicy(state)
        legal = api.legalActions(state)
        self.cost += 1
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        return api.makeMove(policy, legal)

    def policyIteration(self, state):
        states = self.mappedStates   
        policy = self.policy

        while True:
            policyUtils=self.evaluatePolicy(state,policy)
            # print("Evaluated Policy: {}".format(policyUtils))
            policy_stable = True
            for s in states:
                tempPolicy = policy[s]
                # print("Starting one step look ahead")
                bestPolicy = self.oneStepLookAhead(state, s, policyUtils)
                # print("Temp Policy: {}".format(tempPolicy))
                # print("Best Policy: {}".format(bestPolicy))
                if tempPolicy != bestPolicy:
                    policy_stable = False
                    policy[s]=bestPolicy
            if policy_stable:
                self.utils=policyUtils
                self.policy=policy
                break

    def evaluatePolicy(self, state, policy):
        states = self.mappedStates
        reward = self.reward
        policyUtils=self.utils

        while True:
            delta = 0
            for state, action in policy.items():
                # print("Evaluating State: {}".format(state))
                # print("Evaluating Action: {}".format(action))
                currentReward = reward[state] 
                next_states = states[state][action]
                # print("Next States: {}".format(next_states))
                # print("Policy Utility: {}".format(policyUtils))
                tempUtil = self.actionProb * (currentReward + self.gamma * (
                    self.actionProb * policyUtils[next_states[0]] + self.otherActionProb *
                    policyUtils[next_states[1]] + self.otherActionProb * policyUtils[next_states[2]]))
                delta = max(delta, np.abs(tempUtil - policyUtils[state]))
                policyUtils[state] = tempUtil
                # print("Policy Utility for this state: {}".format(policyUtils[state]))
            if delta < self.error:
                break
        return policyUtils

    def oneStepLookAhead(self, state, coord, utils):
        # choose a direction which returns a maximum expected utility
        (x, y) = coord
        walls = api.walls(state) 
        north = (x, y+1)
        south = (x, y-1)
        east = (x+1, y)
        west = (x-1, y)

        # check if the next location is a wall, if its a wall, stop; otherwise, go
        if north in walls:
            north = coord
        if south in walls:
            south = coord
        if east in walls:
            east = coord
        if west in walls:
            west = coord

        # calculate expected utility without reward, since reward is the same
        North_EU = self.actionProb * utils[north] + \
                   self.otherActionProb * utils[west] + \
                   self.otherActionProb * utils[east]
        South_EU = self.actionProb * utils[south] + \
                   self.otherActionProb * utils[west] + \
                   self.otherActionProb * utils[east]
        East_EU = self.actionProb * utils[east] + \
                  self.otherActionProb * utils[north] + \
                  self.otherActionProb * utils[south]
        West_EU = self.actionProb * utils[west] + \
                  self.otherActionProb * utils[north] + \
                  self.otherActionProb * utils[south]

        # get the index of max expected utility
        list = [North_EU, South_EU, East_EU, West_EU]
        maxIndex = 0
        for i in range(len(list)):
            if list[i] > list[maxIndex]:
                maxIndex = i

        # return direction: 0-north  1-south  2-east  3-west
        if maxIndex == 0:
            return "north"
        if maxIndex == 1:
            return "south"
        if maxIndex == 2:
            return "east"
        if maxIndex == 3:
            return "west"

    def getPolicy(self, state):
        # choose a direction which returns a maximum expected utility
        pacman = api.whereAmI(state)
        policy = self.policy
        
        if policy[pacman] == "north":
            return Directions.NORTH
        if policy[pacman] == "south":
            return Directions.SOUTH
        if policy[pacman] == "east":
            return Directions.EAST
        if policy[pacman] == "west":
            return Directions.WEST
        
    def initializePolicy(self, state):
        policy = {}
        for s in self.mappedStates:
            policy[s] = "west"
        self.policy = policy

    def initializeReward(self, state):
        """ Initialize reward for every state """

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

    def checkLastFood(self, state):
        """ check if there is only one piece of food """

        foods = api.food(state)
        if len(foods) == 1:
            return True
        else:
            return False        

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