"""
In search.py, you will implement generic search algorithms
"""

import util
from collections import deque
from queue import PriorityQueue


class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        util.raiseNotDefined()

    def is_goal_state(self, state):
        """
        state: Search state

        Returns True if and only if the state is a valid goal state
        """
        util.raiseNotDefined()

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        util.raiseNotDefined()

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        util.raiseNotDefined()


class GraphNode:
    originator_node = None
    state = None
    move = None
    cost = 0
    heuristic_cost = 1
    cost_so_far = 0

    def __init__(self, origin_node, state, move, cost, heuristic_cost=1, cost_so_far=0):
        self.originator_node = origin_node
        self.state = state
        self.move = move
        self.cost = cost
        self.heuristic_cost = heuristic_cost
        self.cost_so_far = cost_so_far

    def get_moves(self):
        # Get steps by going up the path to origin
        steps = []
        curr_node = self
        while curr_node.originator_node is not None:
            steps.append(curr_node.move)
            curr_node = curr_node.originator_node
        steps.reverse()
        return steps

    def overall_cost(self):
        return self.heuristic_cost + self.cost_so_far

    def __eq__(self, other):
        return self.overall_cost() == other.overall_cost()

    def __gt__(self, other):
        return self.overall_cost() > other.overall_cost()

    def __lt__(self, other):
        return self.overall_cost() < other.overall_cost()


def graph_search_pattern(problem, insertion_func):
    fringe = deque()
    visited_list = set()
    start_state = problem.get_start_state()
    start_node = GraphNode(None, start_state, None, None)
    steps = []

    legal_action_triplets = problem.get_successors(start_state)

    insertion_func(legal_action_triplets, fringe, start_node)
    visited_list.add(start_state)
    while fringe:
        node = fringe.popleft()
        if node.state in visited_list:
            continue
        else:
            visited_list.add(node.state)
        if problem.is_goal_state(node.state):
            steps = node.get_moves()
            break
        else:
            legal_action_triplets = problem.get_successors(node.state)
            insertion_func(legal_action_triplets, fringe, node)

    return steps


def generic_search_pattern(problem, insertion_func):
    fringe = deque()
    visited_list = set()
    start_state = problem.get_start_state()
    steps = None

    legal_action_triplets = problem.get_successors(start_state)
    insertion_func(fringe, legal_action_triplets, [])
    visited_list.add(start_state)
    while fringe:
        curr_actions, triplet = fringe.popleft()
        if triplet[0] in visited_list:
            continue
        else:
            visited_list.add(triplet[0])
        if problem.is_goal_state(triplet[0]):
            curr_actions.append(triplet[1])
            steps = curr_actions
            break
        else:
            curr_actions.append(triplet[1])
            legal_action_triplets = problem.get_successors(triplet[0])
            insertion_func(fringe, legal_action_triplets, curr_actions)

    return steps


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state().state)
    print("Is the start a goal?", problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:", problem.get_successors(problem.get_start_state()))
    """

    def dfs_graph_insertion_func(legal_action_triplets, fringe, curr_node):
        for triplet in legal_action_triplets:
            fringe.insert(0, GraphNode(curr_node, triplet[0], triplet[1], triplet[2]))

    def dfs_insertion_func(fringe, legal_action_triplets, prev_actions):
        for triplet in legal_action_triplets:
            fringe.insert(0, (prev_actions.copy(), triplet))

    return graph_search_pattern(problem, dfs_graph_insertion_func)


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """

    def bfs_graph_insertion_func(legal_action_triplets, fringe, curr_node):
        for triplet in legal_action_triplets:
            fringe.append(GraphNode(curr_node, triplet[0], triplet[1], triplet[2]))

    def bfs_insertion_func(fringe, legal_action_triplets, prev_actions):
        for triplet in legal_action_triplets:
            fringe.insert(-1, (prev_actions.copy(), triplet))

    return graph_search_pattern(problem, bfs_graph_insertion_func)


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def graph_search_pattern_for_a_star(fringe, problem, heuristic=null_heuristic):
    visited_list = set()
    start_state = problem.get_start_state()
    start_node = GraphNode(None, start_state, None, None, heuristic(start_state, problem))
    steps = []

    legal_action_triplets = problem.get_successors(start_state)

    for triplet in legal_action_triplets:
        fringe.put(GraphNode(start_node, triplet[0], triplet[1], triplet[2],
                             heuristic_cost=heuristic(start_state, problem), cost_so_far=triplet[2]))
    visited_list.add(start_state)
    while fringe:
        node = fringe.get()
        if node.state in visited_list:
            continue
        else:
            visited_list.add(node.state)
        if problem.is_goal_state(node.state):
            steps = node.get_moves()
            break
        else:
            legal_action_triplets = problem.get_successors(node.state)
            for triplet in legal_action_triplets:
                fringe.put(GraphNode(node, triplet[0], triplet[1], triplet[2],
                                     heuristic_cost=heuristic(start_state, problem),
                                     cost_so_far=triplet[2] + node.cost_so_far))

    return steps


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    fringe_queue = PriorityQueue()
    return graph_search_pattern_for_a_star(fringe_queue, problem, heuristic)


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
