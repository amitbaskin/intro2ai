"""
In search.py, you will implement generic search algorithms
"""

import util
from collections import deque


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


def generic_search_pattern(problem, insertion_func):
    def search(current_state):
        legal_action_triplets = problem.get_successors(current_state)
        insertion_func(fringe, legal_action_triplets, actions)
        while fringe:
            curr_actions, triplet = fringe.popleft()
            if triplet[0] in visited_list:
                continue
            else:
                visited_list.add(triplet[0])
            if problem.is_goal_state(triplet[0]):
                curr_actions.append(triplet[1])
                return True, curr_actions
            else:
                curr_actions.append(triplet[1])
                legal_action_triplets = problem.get_successors(triplet[0])
                insertion_func(fringe, legal_action_triplets, curr_actions)

    fringe = deque()
    actions = []  # List of steps we went through
    visited_list = set()
    start_state = problem.get_start_state()
    visited_list.add(start_state)
    _, steps = search(start_state)
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
    def dfs_insertion_func(fringe, legal_action_triplets, prev_actions):
        for triplet in legal_action_triplets:
            fringe.insert(-1, (prev_actions.copy(), triplet))

    return generic_search_pattern(problem, dfs_insertion_func)


def breadth_first_search(problem):
    """
    Search the shallowest nodes in the search tree first.
    """
    def bfs_insertion_func(fringe, legal_action_triplets, prev_actions):
        for triplet in legal_action_triplets:
            fringe.insert(0, (prev_actions.copy(), triplet))

    return generic_search_pattern(problem, bfs_insertion_func)


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


def a_star_search(problem, heuristic=null_heuristic):
    """
    Search the node that has the lowest combined cost and heuristic first.
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()



# Abbreviations
bfs = breadth_first_search
dfs = depth_first_search
astar = a_star_search
ucs = uniform_cost_search
