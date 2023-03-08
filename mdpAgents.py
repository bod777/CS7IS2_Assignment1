# mdpAgents.py
# Written by Hana Mizukami using provided CW template from keats
# December 1st 2021
#
# Version 9
#
# Intended to work with the PacMan AI projects from:
# http://ai.berkeley.edu/
#
# These use a simple API that allow us to control Pacman's interaction with
# the environment adding a layer on top of the AI Berkeley code.
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# The agent here is was written by Simon Parsons, based on the code in
# pacmanAgents.py
#

# I have used reading resources such as https://towardsdatascience.com/reinforced-pac-man-8e51409f4fc to understand Markov Decision Processes behind Pacman, 
# https://towardsdatascience.com/introduction-to-reinforcement-learning-markov-decision-process-44c533ebf8da to understand reinforcement learning, 
# https://www.lesswrong.com/posts/bG4PR9uSsZqHg2gYY/utility-reward, and 
# https://medium.com/@m.alzantot/deep-reinforcement-learning-demysitifed-episode-2-policy-iteration-value-iteration-and-q-978f9e89ddaa to use value iteration.
# I have also used the Grid class from Practical 5 to check the pacman game from keats https://keats.kcl.ac.uk/mod/resource/view.php?id=5173681.
# I followed https://www.python.org/dev/peps/pep-0008/#names-to-avoid for coding conventions of Python.
#
#
# I illustrated how the code works in an image: https://drive.google.com/drive/folders/1Rt8zuGGghJKD3speGgSZQKcQOa99-yhh?usp=sharing
from pacman import Directions
from game_modules.game import Agent
import api
import random
import game_modules.game as game
import game_modules.util as util

# From mapAgents.py of Week 5 Practical Solutions Keats https://keats.kcl.ac.uk/mod/resource/view.php?id=5173681
# 
class Grid:       
    # Constructor
    # grid:   an array that has one position for each element in the grid.
    # width:  the width of the grid
    # height: the height of the grid
    #
    def __init__(self, width, height):
        self.width = width
        self.height = height
        subgrid = []
        for i in range(self.height):
            row=[]
            for j in range(self.width):
                row.append(0)
            subgrid.append(row)
        self.grid = subgrid

    # Print the grid out.
    def display(self):       
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print(self.grid[i][j],)
            # A new line after each line of the grid
            print 
        # A line after the grid
        print

    # The display function prints the grid out upside down. This
    # prints the grid out so that it matches the view we see when we
    # look at Pacman.
    def prettyDisplay(self):       
        for i in range(self.height):
            for j in range(self.width):
                # print grid elements with no newline
                print(self.grid[self.height - (i + 1)][j],)
            # A new line after each line of the grid
            print 
        # A line after the grid
        print
        
    # Set and get the values of specific elements in the grid.
    # Here x and y are indices.
    def setValue(self, x, y, value):
        self.grid[y][x] = value

    def getValue(self, x, y):
        return self.grid[y][x]

    # Return width and height to support functions that manipulate the
    # values stored in the grid.
    def getHeight(self):
        return self.height

    def getWidth(self):
        return self.width

'''MDP Definition from https://towardsdatascience.com/understanding-the-markov-decision-process-mdp-8f838510f150
    (S, A, P, R, gamma) S: states, A: actions, P: state-transition probability, R: reward, gamma: discount factor 
    Introduction to actions elicits a notion of control over the Markov Process, i.e., previously, the state transition probability and the state rewards were more or less stochastic (random). 
    However, now the rewards and the next state also depend on what action the agent picks. 
    Basically, the agent can now control its own fate (to some extent).'''

#constants of rewards, discount factor and the number of iterations
FOOD_REWARD = 1.05
GHOST_REWARD = -12
DISCOUNT_FACTOR = 0.92 
ITERATION_NUMBER = 40
EMPTY_VALUE = -0.04

class MDPAgent(Agent):

    # Constructor: this gets run when we first invoke pacman.py
    def __init__(self):
        self.food_re = FOOD_REWARD # initial value for food reward for any box that has a food
        self.empty_val = EMPTY_VALUE #  initial value for empty boxes
        self.ghost_re = GHOST_REWARD #  initial value for ghosts for any box that has a ghost
        self.ghost_friend = 0 # variable for ghost that is edible/friendly
        self.expected_utilities = [] # list of all expected utilities
        self.expected_location = [] # list of coordinates for the above expected utilities
        self.gamma = DISCOUNT_FACTOR # gamma : https://towardsdatascience.com/the-bellman-equation-59258a0d3fa7

    # Gets run after an MDPAgent object is created and assigns initial state and costs (initial state of the value iteration process)
    def registerInitialState(self, state):
        print ("Running registerInitialState for MDPAgent!")
        print ("I'm at:")
        print (api.whereAmI(state))
        self.makeMap(state) #from keats
        self.addWallsToMap(state) #from keats
        self.updateFoodInMap(state) #from keats
        self.map.display() #fromkeats
        MDPAgent.reapplyInitialValues(self, self.food_re, 0)

    # When the game ends
    def final(self, state):
        '''sets the values back to the original values after ending games'''
        print ("Looks like the game just ended!")
        self.food_re = FOOD_REWARD 
        self.empty_val = EMPTY_VALUE
        self.ghost_re = GHOST_REWARD
        self.ghost_friend = 0
        self.expected_utilities = []
        self.expected_location = []
        self.gamma = DISCOUNT_FACTOR
        MDPAgent.reapplyInitialValues(self, self.food_re, 0)

    # Make a map by creating a grid of the right size
    # From Keats Week 5 Practical MapAgent
    def makeMap(self,state):
        corners = api.corners(state)
        print(corners)
        height = self.getLayoutHeight(corners)
        width  = self.getLayoutWidth(corners)
        self.map = Grid(width, height)
        
    # Functions to get the height and the width of the grid.
    #
    # We add one to the value returned by corners to switch from the
    # index (returned by corners) to the size of the grid
    # From Keats Week 5 Practical MapAgent
    def getLayoutHeight(self, corners):
        height = -1
        for i in range(len(corners)):
            if corners[i][1] > height:
                height = corners[i][1]
        return height + 1

    def getLayoutWidth(self, corners):
        width = -1
        for i in range(len(corners)):
            if corners[i][0] > width:
                width = corners[i][0]
        return width + 1

    # Functions to manipulate the map.
    #
    # Put every element in the list of wall elements into the map
    # From Keats Week 5 Practical MapAgent
    def addWallsToMap(self, state):
        walls = api.walls(state)
        for i in range(len(walls)):
            self.map.setValue(walls[i][0], walls[i][1], '%')

    # Create a map with a current picture of the food that exists.
    # From Keats Week 5 Practical MapAgent
    def updateFoodInMap(self, state):
        # First, make all grid elements that aren't walls blank.
        for i in range(self.map.getWidth()):
            for j in range(self.map.getHeight()):
                if self.map.getValue(i, j) != '%':
                    self.map.setValue(i, j, ' ')
        food = api.food(state)
        for i in range(len(food)):
            self.map.setValue(food[i][0], food[i][1], '*')

    # Functions to get ghosts states
    # From Keats Week 2 Practical SampleAgents
    def getGhost(self, state):
        ghosts = api.ghost(state) 
        return ghosts
    
    # Function to get the type of box type: empty, food, wall
    # @param self: MDP Agent
    def getBoxType(self):
        boxType = 0
        for i in range(self.map.getHeight()):
            for j in range(self.map.getWidth()): 
                if self.map.getValue(j,i) == " ":
                    boxType = 0
                elif self.map.getValue(j,i) == "*":
                    boxType = 1
                elif self.map.getValue(j,i) == "%":
                    boxType = 2
        return boxType

    # Function to reapply initial state values after the game ends 
    #
    # Goes through the grid and applies the given food reward and empty value 
    # @param self: MDP Agent
    # @param food_re: given food reward
    # @type food_re: float
    # @param emp_reward: given empty box value
    # @type emp_reward: float
    def reapplyInitialValues(self, food_re, emp_reward): 
        for i in range(self.map.getHeight()):
            for j in range(self.map.getWidth()): 
                if self.map.getValue(j,i) == " ": #if the location type is an empty box
                    self.expected_utilities.append(emp_reward) #add empty box value
                    self.expected_location.append((j,i))
                elif self.map.getValue(j,i) == "*" : #if the location contains food
                    self.expected_utilities.append(food_re) #add food reward
                    self.expected_location.append((j,i))

    # Function to apply value iteration to the state and store return expected utilities in a list
    #
    # @param self: MDP Agent
    # @param state: state of the agent
    # @return neighboring expected utilities
    # @returntype: list 
    def calculateValueIte(self,state):
        surrounding_ex_util = [] # stores neighboring expected utilities 

        for i in range(len(self.expected_utilities)): # iterate all expected utilities 
            west_val = self.map.getValue(self.expected_location[i][0]-1, self.expected_location[i][1]) #retrieves expected utility value from expected_location for west
            west_loc = (self.expected_location[i][0]-1, self.expected_location[i][1])
            west = MDPAgent.getWestUtility(self, west_val, self.expected_utilities[i], west_loc)

            east_val = self.map.getValue(self.expected_location[i][0]+1, self.expected_location[i][1])
            east_loc = (self.expected_location[i][0]+1, self.expected_location[i][1])
            east = MDPAgent.getEastUtility(self, east_val, self.expected_utilities[i], east_loc)

            north_val = self.map.getValue(self.expected_location[i][0], self.expected_location[i][1]+1)
            north_loc = (self.expected_location[i][0], self.expected_location[i][1]+1)
            north = MDPAgent.getNorthUtility(self, north_val, self.expected_utilities[i], north_loc)
            
            south_val = self.map.getValue(self.expected_location[i][0], self.expected_location[i][1]-1)
            south_loc= (self.expected_location[i][0], self.expected_location[i][1]-1)
            south = MDPAgent.getSouthUtility(self, south_val, self.expected_utilities[i], south_loc)

            # westval = west * 0.8 + (north + south) * 0.1
            # the transition model is such that when the agent tries to move in a given direction, 10% of the time it moves to the left of the chosen direction1, and 10% of the time it moves to the right.
            left_value = 0.8 * west + (south + north) * 0.1

            right_value = 0.8 * east + (south + north) * 0.1

            down_value = 0.8 * south + (east + west) * 0.1

            up_value = 0.8 * north + (east + west) * 0.1

            # finds the maximum value from all directions
            max_val = max([(up_value), (left_value), (down_value), (right_value)])           
            
            # Bellman optimality equality equation to update the value of state from the value of other states that could be potentially reached from state http://incompleteideas.net/book/bookdraft2018jan1.pdf
            bellman_val = self.empty_val + (self.gamma * max_val)
           
            # # gets ghosts states 
            # ghosts = api.ghosts(state) 
            # print(ghosts)
            # ghost_number = len(ghosts) # Number of ghosts
            # #print (ghost_number)
            # # From Practical 2 SearchingAgent
            # #print ("Distance to ghosts:")
            #    # for i in range(len(theGhosts)):
            # #print (util.manhattanDistance(self,theGhosts[i])) 

            # # check if the location of the expected utility is at a food or an empty space
            # if self.map.getValue(self.expected_location[i][0], self.expected_location[i][1]) == "*" or self.map.getValue(self.expected_location[i][0], self.expected_location[i][1]) == " ":
            #     # if ghosts are at the same state (first comparison for smallGrid second comparison for mediumGrid)
            #     if (ghosts[0][0], ghosts[0][1]) == (self.expected_location[i][0], self.expected_location[i][1]) or (ghosts[ghost_number-1][0][0], ghosts[ghost_number-1][0][1]) == (self.expected_location[i][0], self.expected_location[i][1]) :
            #         #print (ghost_number)
            #         if ghosts[0][1] == 1 and ghosts[ghost_number-1][1] == 1:
            #             # add ghost_friend value to the neighboring expected utility if it is edible
            #             surrounding_ex_util.append(self.ghost_friend)
            #         else :
            #             # add ghost value to the neighboring expected utility if it is not edible/friendly
            #             surrounding_ex_util.append(self.ghost_re)
            #     else :  # if ghosts are not at the same state
            #         if ghost_number == 2 : #for mediumGrid 
            #             # determine distance to check non-ghost boxes 
            #             # using manhattanDistance from Practical 2, 3 Keats
            #             ghost_distance_1 = util.manhattanDistance(self.expected_location[i],ghosts[0][0])
            #             ghost_distance_2 = util.manhattanDistance(self.expected_location[i],ghosts[1][0])
            #         else : #for smallGrid make the second ghost "disappear"
            #             ghost_distance_1 = util.manhattanDistance(self.expected_location[i],ghosts[0][0])
            #             ghost_distance_2 = 10000
                    
            #         #if the searched location is at a food
            #         if self.map.getValue(self.expected_location[i][0], self.expected_location[i][1]) == "*" :
            #             if self.map.getHeight() <= 7: # for small grid
            #                 if ghost_distance_1 <= 2 or ghost_distance_2 <= 2 : #considering the capacity of the pacman to stay away from the ghost
            #                     surrounding_ex_util.append(bellman_val)  # add calculated utility when the ghost is nearby
            #                 else : 
            #                     surrounding_ex_util.append(self.food_re)  # add food value to the neighboring expected utility  
            #             elif self.map.getHeight() > 7: # for medium grid
            #                 if ghost_distance_1 <= 5 or ghost_distance_2 <= 5 : #considering the capacity of the pacman distinguish the small grid and medium grid
            #                     surrounding_ex_util.append(bellman_val)  # add calculated utility when the ghost is nearby   
            #                 else : 
            #                     surrounding_ex_util.append(self.food_re)  # add food value to the neighboring expected utility 
            #         else :  # if it is next to an empty box
            #             surrounding_ex_util.append(bellman_val)  
        return surrounding_ex_util # return neighboring expected utilities 
    
    # Function gets the value to calculate the west of pacman's utility 
    # Using Tutorial 4's utility calculation model for action to go to west
    # @param self: MDP Agent
    # @param stateType: the neighboring space's type (wall, empty space, food) from the map
    # @type stateType: String
    # @param current_exp: expected utility of the current state
    # @type current_exp: float
    # @param west_coordinates: x, y coordinates of the pacman's location to the west
    # @type west_coordinates: tuple 
    def getWestUtility(self, stateType, current_exp, west_coordinates):
        exp_val = 0 # stores utility value
        if stateType == "%" : # if it is next to the wall
            exp_val = current_exp  #expected utility value does not change
        elif stateType == " " or stateType == "*": # if it is next to a space or food
                exp_val = self.expected_utilities[self.expected_location.index(west_coordinates)] # retrieves neighboring expected utility from the coordinates
        return exp_val

    # Function gets the value to calculate the west of pacman's utility 
    # Using Tutorial 4's utility calculation model for action to go to east
    # @param self: MDP Agent
    # @param stateType: the neighboring space's type (wall, empty space, food) from the map
    # @type stateType: String
    # @param current_exp: expected utility of the current state
    # @type current_exp: float
    # @param west_coordinates: x, y coordinates of the pacman's location to the east
    # @type west_coordinates: tuple 
    def getEastUtility(self, stateType, current_exp, east_location):
        exp_val = 0 # stores utility value
        if stateType == "%" : # if it is next to the wall
            exp_val = current_exp #expected utility value does not change
        elif stateType == " " or stateType == "*": # if it is next to a space or food
                exp_val = self.expected_utilities[self.expected_location.index(east_location)] # retrieves neighboring expected utility from the coordinates
        return exp_val

    # Function gets the value to calculate the west of pacman's utility 
    # Using Tutorial 4's utility calculation model for action to go to north
    # parameters and types same as west and east
    def getNorthUtility(self, stateType, current_exp, north_location):
        exp_val = 0 # stores utility value
        if stateType == "%" : # if it is next to the wall
            exp_val = current_exp #expected utility value does not change
        elif stateType == " " or stateType == "*": # if it is next to a space or food
                exp_val = self.expected_utilities[self.expected_location.index(north_location)] # retrieves neighboring expected utility from the coordinates
        return exp_val
    
    # Function gets the value to calculate the west of pacman's utility 
    # Using Tutorial 4's utility calculation model for action to go to north
    # parameters and types same as west and east
    def getSouthUtility(self, stateType, current_exp, south_location):
        exp_val = 0 # stores utility value
        if stateType == "%" : # if it is next to the wall
            exp_val = current_exp #expected utility value does not change
        elif stateType == " " or stateType == "*": # if it is next to a space or food
                exp_val = self.expected_utilities[self.expected_location.index(south_location)] # retrieves neighboring expected utility from the coordinates
        return exp_val

    # Function to get values for each move action
    # @param legal: MDPAgent legal moves
    # @param pacman: MDPAgent
    # @param hori_coor: horizontal coordinates of the pacman
    # @param verti_coor: vertical coordinates of the pacman
    # @return: values from the actions
    # @returntype: list
    def getActionValues(legal, pacman, hori_coor, verti_coor):
        values = []
        for action in legal:
            value = None
            if action == Directions.NORTH:
                value = pacman[verti_coor + 1][hori_coor]
            elif action == Directions.SOUTH:
                value = pacman[verti_coor - 1][hori_coor]
            elif action == Directions.EAST:
                value = pacman[verti_coor][hori_coor + 1]
            elif action == Directions.WEST:
                value = pacman[verti_coor][hori_coor - 1]
        return values


    def getAction(self, state):
        self.updateFoodInMap(state)     
        # Remove STOP as a legal action as it will not be a necessary action at any state during the game
        legal = api.legalActions(state)
        pacman = api.whereAmI(state)
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)
        #print
        self.iteration = 0 # counts how many times it iterates through the value iteration and sets limit
        
        while (self.iteration < 40) : # iterate until 40 times
            self.expected_utilities = MDPAgent.calculateValueIte(self,state)  # calculate Value Iteration for the next moves
            self.iteration += 1 # increment iteration count to limit
        
        exp_util_legal = [] # list of expected utilities based on the moves
        moves = [] # list of moves to search through the maximum
        for i in range(len(legal)):  # Goes through all legal moves
            if legal[i] == "West":
                left_coor = (pacman[0]-1, pacman[1])  # variable to set the coordinates west to the pacman
                exp_util_legal.append(self.expected_utilities[self.expected_location.index(left_coor)]) #from the coordinates add the expected utility to the list
                moves.append("West") # moves contains the West as a move
            elif legal[i] == "East":
                right_coor = (pacman[0]+1, pacman[1]) # variable to set the coordinates east to the pacman
                exp_util_legal.append(self.expected_utilities[self.expected_location.index(right_coor)]) #from the coordinates add the expected utility to the list
                moves.append("East") # moves contains the East as a move
            elif legal[i] == "North":
                up_coor = (pacman[0], pacman[1]+1) # variable to set the coordinates north to the pacman
                exp_util_legal.append(self.expected_utilities[self.expected_location.index(up_coor)]) #from the coordinates add the expected utility to the list
                moves.append("North") # moves contains the North as a move
            elif legal[i] == "South":
                down_coor = (pacman[0], pacman[1]-1) # variable to set the coordinates south to the pacman
                exp_util_legal.append(self.expected_utilities[self.expected_location.index(down_coor)]) #from the coordinates add the expected utility to the list
                moves.append("South") # moves contains the South as a move
        
        optimal_move = moves[exp_util_legal.index(max(exp_util_legal))] #finds the move that has the maximum value of expected utility in the neighboring space  and stores as a move
        if optimal_move in legal :
            return api.makeMove(optimal_move, legal) # pacman moves to the space with the maximum expected utility