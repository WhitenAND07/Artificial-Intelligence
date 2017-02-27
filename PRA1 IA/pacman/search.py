# search.py


import util
import node
import sys

class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).
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

    return  [s, s, w, s, w, w, s, w]

def depthFirstSearch(problem):
    """
    Search the deepest nodes in the search tree first.
    Your search algorithm needs to return a list of actions that reaches the
    goal. Make sure to implement a graph search algorithm.
    """
    fringe = [node.Node(problem.getStartState())]
    while True:

    	if len(fringe) == 0:
        	print "No solution"
        	sys.exit(-1)

	n = fringe.pop()

	if problem.isGoalState(n.state):
        	return n.path()
	for s, a, c in problem.getSuccessors(n.state):
		fringe.append(node.Node(s, n, a, c))

    util.raiseNotDefined()


def breadthFirstSearch(problem):
    """Search the shallowest nodes in the search tree first."""
    fringe = [node.Node(problem.getStartState())]
    generated = {}

    while True:

    	if len(fringe) == 0:
        	print "No solution" #Para mapas sin solucion
        	sys.exit(-1)

	n = fringe.pop(0)
	generated[n.state] = []

	if problem.isGoalState(n.state):
        	return n.path()
		
	for s, a, c in problem.getSuccessors(n.state):

		if s not in generated:
			fringe.append(node.Node(s, n, a, c))
			#generated[s] = 

    util.raiseNotDefined()

def uniformCostSearch(problem):
    """Search the node of least total cost first."""
    return customSearch(problem)

    util.raiseNotDefined()

def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0

def aStarSearch(problem, heuristic=nullHeuristic):
    """Search the node that has the lowest combined cost and heuristic first."""
    return customSearch(problem, heuristic)

    util.raiseNotDefined()

def bestHFirst(problem, heuristic):

    return customSearch(problem,heuristic,0)

    util.raiseNotDefined()

def bidirectionalSearch(problem):
    fringeStart = util.Queue()  # Fringe for startState()
    fringeEnd = util.Queue()  # Fringe for goal (1, 1)

    fringeStart.push(node.Node(problem.getStartState()))
    fringeEnd.push(node.Node((1, 1)))

    generatedStart = dict()  # Start point generated for startState()
    generatedEnd = dict()  # Point generated for the target (1,1)

    while not fringeStart.isEmpty() and not fringeEnd.isEmpty():

        na = fringeStart.pop()
        nb = fringeEnd.pop()

        generatedStart[na.state] = (na, 'E')
        generatedEnd[nb.state] = (nb, 'E')

        if generatedStart.get(na.state)[1] != 'F':
            path = binaryExpand(generatedStart, generatedEnd,
                                fringeStart, na, problem, 'S')

            if path is not None:
                return path

        if generatedEnd.get(nb.state)[1] != 'F':
            path = binaryExpand(generatedEnd, generatedStart,
                            fringeEnd, nb, problem, 'E')

            if path is not None:
                return path

    return sys.exit(-1)

def binaryExpand(actualGenerated, oppositeGenerated, fringe, pn, problem, pos):

    for state, action, cost in problem.getSuccessors(pn.state):
        ns = node.Node(state, pn, action, cost)

        if ns.state in oppositeGenerated:
            if pos == 'S':
                path = reversePath(oppositeGenerated.get(ns.state)[0])
                return ns.path() + path
            else:
                path = reversePath(ns)
                return oppositeGenerated.get(ns.state)[0].path() + path

        if ns.state not in actualGenerated:
            actualGenerated[ns.state] = (ns, 'F')
            fringe.push(ns)

def reversePath(node):

    path = node.path()
    path.reverse()
    pathReversed = []

    for p in path:
        if p == 'East':
            pathReversed.append('West')
        elif p == 'West':
            pathReversed.append('East')
        elif p == 'North':
            pathReversed.append('South')
        else:
            pathReversed.append('North')

    return pathReversed

def customSearch(problem, heuristic=nullHeuristic, c=1):

    """ 
    UCS, Astar, Best-Heurist First
        UCS = Null Heuristic
	    Astar = Heuristic + Cost
	    Best-Heuristic First = Cost 0
    """

    fringe = util.PriorityQueue()
    fringe.push(node.Node(problem.getStartState()),0)
    expanded = []

    while not fringe.isEmpty():
        n = fringe.pop()

        if problem.isGoalState(n.state):
            return n.path()

        expanded.append(n.state)

        for s, a, t in problem.getSuccessors(n.state):
            new_h = max(heuristic(s,problem) + n.cost + c,\
                        heuristic(n.state,problem) + n.cost) #PathMax

	    if s in fringe.heap or s not in expanded:
                fringe.update(node.Node(s, n, a, c + n.cost),new_h)

    util.raiseNotDefined()

# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
bfsh = bestHFirst
bds = bidirectionalSearch