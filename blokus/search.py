"""
In search.py, you will implement generic search algorithms
"""

import util
import numpy as np


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

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves
        """
        util.raiseNotDefined()


class GraphNode:
    originator_node = None
    state = None
    move = None
    cost = 0
    heuristic_cost = 1
    cost_so_far = 0
    a_star_cost = 0

    def __init__(self, origin_node, state, move, cost, heuristic_cost=1):
        self.originator_node = origin_node
        self.state = state
        self.move = move
        self.cost = cost
        self.heuristic_cost = heuristic_cost
        self.cost_so_far = self.originator_node.cost_so_far + cost if \
            self.originator_node is not None else cost
        self.a_star_cost = self.cost_so_far + heuristic_cost

    def __str__(self):
        return str(self.state.state)

    def get_moves(self):
        steps = []
        curr_node = self
        while curr_node.originator_node is not None:
            steps.append(curr_node.move)
            curr_node = curr_node.originator_node
        steps.reverse()
        return steps

    def total_cost(self):
        return self.heuristic_cost + self.cost_so_far

    def __eq__(self, other):
        return self.total_cost() == other.total_cost()

    def __gt__(self, other):
        return self.total_cost() > other.total_cost()

    def __lt__(self, other):
        return self.total_cost() < other.total_cost()


def null_heuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the  current state to the
    nearest goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0


def generic_cost_func(node):
    return node.cost


def generic_check_func(node, visited_list):
    node_str = node.__str__()

    return (node_str in visited_list.keys()) and (visited_list[node_str] <=
                                                  node.a_star_cost)


def generic_insertion_func(legal_action_triplets, curr_fringe, curr_node):
    for triplet in legal_action_triplets:
        new_node = GraphNode(curr_node, triplet[0], triplet[1], triplet[2])
        if new_node.heuristic_cost == np.inf:
            continue
        curr_fringe.push(new_node)


def graph_search_pattern(fringe, problem, insertion_func, check_func,
                         cost_func, heuristic=null_heuristic):
    visited_list = dict()
    start_state = problem.get_start_state()
    start_node = GraphNode(None, start_state, None, 0,
                           heuristic(start_state, problem))
    cur_successors = problem.get_successors(start_node.state)
    insertion_func(cur_successors, fringe, start_node)
    visited_list[start_node.__str__()] = 0
    while fringe:
        node = fringe.pop()
        if check_func(node, visited_list):
            continue
        else:
            visited_list[node.__str__()] = cost_func(node)
        if problem.is_goal_state(node.state):
            return node.get_moves()
        else:
            cur_successors = problem.get_successors(node.state)
            insertion_func(cur_successors, fringe, node)
    print('no solution')
    exit()


def depth_first_search(problem):
    """
    Search the deepest nodes in the search tree first.

    Your search algorithm needs to return a list of actions that reaches
    the goal. Make sure to implement a graph search algorithm.

    To get started, you might want to try some of these simple commands to
    understand the search problem that is being passed in:

    print("Start:", problem.get_start_state().state)
    print("Is the start a goal?",
    problem.is_goal_state(problem.get_start_state()))
    print("Start's successors:",
    problem.get_successors(problem.get_start_state()))
    """
    fringe = util.Stack()
    return graph_search_pattern(fringe, problem, generic_insertion_func,
                                generic_check_func, generic_cost_func)


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    fringe = util.Queue()
    return graph_search_pattern(fringe, problem, generic_insertion_func,
                                generic_check_func, generic_cost_func)


def uniform_cost_search(problem):
    """
    Search the node of least total cost first.
    """
    def ucs_cost_func(node):
        return node.cost_so_far

    fringe_queue = util.PriorityQueueWithFunction(lambda x: x.cost_so_far)
    return graph_search_pattern(fringe_queue, problem,
                                generic_insertion_func, generic_check_func,
                                ucs_cost_func)


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    def a_star_cost_func(node):
        return node.a_star_cost

    fringe_queue = util.PriorityQueueWithFunction(lambda x: x.a_star_cost)
    return graph_search_pattern(fringe_queue, problem, generic_insertion_func,
                                generic_check_func, a_star_cost_func, heuristic)


# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
