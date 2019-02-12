# search.py
# ---------
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


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""

import util


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


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions

    s = Directions.SOUTH
    w = Directions.WEST
    return [s, s, w, s, w, w, s, w]


def generalSearchAlgorithm(problem, container, heuristic=None):
    """
    A generalized search algorithm with a storage container for
    better selection of nodes.
    :param problem: Problem grid with nodes
    :param container: How to prioritize what nodes to search next
    :param heuristic: Heuristics, if applicable
    :return list of actions to take
    """
    # What nodes have we already been to
    nodes_visited = []

    # Container of nodes to still traverse
    # Add starting node to the container
    if heuristic:
        container.push((problem.getStartState(), [], 0), 0)
    else:
        container.push((problem.getStartState(), []))

    while not container.isEmpty():
        # Get the next element's components, based on the container type
        if heuristic:
            node, node_actions, node_cost = container.pop()
        else:
            node, node_actions = container.pop()
            node_cost = -1

        # If we've visited it already, skip it
        if node not in nodes_visited:
            # Record we've been here already
            nodes_visited.append(node)

            # See if we're done
            if problem.isGoalState(node):
                return node_actions

            # Since we're not done, add the new selections to the container
            for next_node, next_action, next_cost in problem.getSuccessors(node):
                if heuristic:
                    container.push(
                        (
                            next_node,
                            node_actions + [next_action],
                            node_cost + next_cost,
                        ),
                        node_cost + next_cost,
                    )
                else:
                    container.push((next_node, node_actions + [next_action]))


def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:
    print "Start:", problem.getStartState()
    print "Is the start a goal?", problem.isGoalState(problem.getStartState())
    print "Start's successors:", problem.getSuccessors(problem.getStartState())
    """
    return generalSearchAlgorithm(problem, util.Stack())


def breadthFirstSearch(problem):
    """
    Search via BFS method (Similar to DFS but uses a Queue instead)
    """
    return generalSearchAlgorithm(problem, util.Queue())


def uniformCostSearch(problem):
    """
    Search the node of least total cost first.
    """
    return generalSearchAlgorithm(problem, util.PriorityQueue(), nullHeuristic)


def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    return generalSearchAlgorithm(problem, util.PriorityQueue(), heuristic)


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
