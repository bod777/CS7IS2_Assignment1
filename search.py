# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.

"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import game_modules.util as util

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()

def depthFirstSearch(problem):
    visited_node = []
    actions = []
    stack = util.Stack()

    # Gets the starting state of the problem.
    start_node = problem.getStartState() 
    # If the starting state is a goal state, return an empty list because no actions need to be taken.
    if problem.isGoalState(start_node):
        return []

    # Adds the starting node and the empty list of actions to the stack.
    stack.push((start_node, actions))

    # While stack is not empty
    while stack:
        # Pop the next node and its corresponding list of actions from the stack.
        node, action = stack.pop()
        if node not in visited_node:
            # Add the node to the list of visited nodes.
            visited_node.append(node)
            # If the node is a goal state, return the list of actions taken to reach the node.
            if problem.isGoalState(node):
                return action
            for successor, direction, cost in problem.getSuccessors(node):
                new_action = action + [direction]
                stack.push((successor, new_action))
    return []

def breadthFirstSearch(problem):
    visited_node = []
    actions = []
    queue = util.Queue()

    # Gets the starting state of the problem.
    start_node = problem.getStartState() 
    # If the starting state is a goal state, return an empty list because no actions need to be taken.
    if problem.isGoalState(start_node):
        return []
    
    # Insert start node into the queue
    queue.push((start_node, actions)) 

    # While quere is not empty
    while queue: 
        # Pop the next node and its corresponding list of actions from the queue.
        node, action = queue.pop() 
        # Process all the neighbors of front node
        if node not in visited_node:
            visited_node.append(node)
            # If the node is a goal state, return the list of actions taken to reach the node.
            if problem.isGoalState(node):
                return action
            for successor, direction, cost in problem.getSuccessors(node):
                new_action = action + [direction] 
                queue.push((successor, new_action))
    return []

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    visited_node = []
    # Gets the starting state of the problem.
    start_node = problem.getStartState() 
    # If the starting state is a goal state, return an empty list because no actions need to be taken.
    if problem.isGoalState(start_node):
        return [] 
    
    queue = util.PriorityQueue()
    # Adds the starting node, the empty list of actions, and a cost of 0 to the priority queue. 
    queue.push((start_node, [], 0), 0)

    # While queue is not empty
    while not queue.isEmpty():
        # Pop the next node, its corresponding list of actions, and its previous cost from the priority queue.
        node, action, prev_cost = queue.pop()
        if node not in visited_node:
            # Add the node to the list of visited nodes.
            visited_node.append(node)
            # If the node is a goal state, return the list of actions taken to reach the node.
            if problem.isGoalState(node):
                return action
            for successor, direction, cost in problem.getSuccessors(node):
                # If the successor has already been visited, skip it.
                if successor in visited_node:
                    continue
                # Create a new list of actions that includes the action taken to get to the successor.
                new_action = action + [direction]
                # Calculate the new cost of getting to the successor by adding the cost of the action taken to the previous cost.
                new_cost = prev_cost + cost
                # Calculate the new heuristic for the successor node by adding the new cost to the estimated cost from the successor
                # to the goal state, as determined by the heuristic function.
                new_heuristic = new_cost + heuristic(successor, problem)
                # Add the successor, the new list of actions, and the new cost to the priority queue, with the priority set to the new heuristic.
                queue.push((successor, new_action, new_cost), new_heuristic)
    return []
