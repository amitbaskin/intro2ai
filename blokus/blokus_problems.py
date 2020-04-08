from board import Board
from search import SearchProblem, ucs
import util
import numpy as np
import math


class BlokusFillProblem(SearchProblem):
    """
    A one-player Blokus game as a search problem.
    This problem is implemented for you. You should NOT change it!
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.expanded = 0

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(state.pieces[0])

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, 1) for move in
                state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves
        """
        return len(actions)


#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
def get_corners(problem):
    return [(0, problem.board_w-1), (problem.board_h-1, 0),
             (problem.board_h-1, problem.board_w-1)]


def get_forbidden_adjacents(problem, pos):
    adjacents = [(pos[0]-1, pos[1]), (pos[0]+1, pos[1]),
                           (pos[0], pos[1]+1), (pos[0], pos[1]-1)]
    forbidden_adjacents = []
    for adjacent in adjacents:
        if problem.board.check_tile_legal(0, adjacent[0], adjacent[1]):
            forbidden_adjacents.append(adjacent)
    return forbidden_adjacents


def get_total_forbidden_positions(state, problem):
    forbidden_positions = []
    for target in get_current_targets(state, problem):
        forbidden_positions += get_forbidden_adjacents(problem, target)
    return forbidden_positions


def is_state_authorized(state, problem):
    forbidden_positions = get_total_forbidden_positions(state, problem)
    for pos in forbidden_positions:
        if state.state.item(pos) != -1:
            return False
    return True


def get_heuristic_cost(state, problem):
    if is_state_authorized(state, problem):
        return get_sum_distances_from_targets(state, problem)
    else:
        return np.inf


def get_distance_between_positions(pos1, pos2):
    x_diff = math.fabs(pos1[0] - pos2[0])
    y_diff = math.fabs(pos1[1] - pos2[1])
    return max(x_diff, y_diff)


def get_current_targets(state, problem):
    current_corners = []
    for target in problem.targets:
        if state.state.item(target) == -1:
            current_corners.append(target)
    return current_corners


def get_min_dist_from_goal(state, problem, goal):
    played_positions = []
    for row_num in range(problem.board_h):
        for column_num in range(problem.board_w):
            cur_pos = (row_num, column_num)
            if state.state.item(cur_pos) != -1:
                played_positions.append((row_num, column_num))
    min_dist = np.inf
    for pos in played_positions:
        cur_dist = get_distance_between_positions(pos, goal)
        if cur_dist < min_dist:
            min_dist = cur_dist
    return min_dist


def get_sum_distances_from_targets(state, problem):
    sm = 0
    cur_goals = get_current_targets(state, problem)
    len_cur_goals = len(cur_goals)
    # print(len_cur_goals)
    # print(state)
    if len_cur_goals == 0:
        return 0
    else:
        for corner in cur_goals:
            sm += get_min_dist_from_goal(state, problem, corner)
        # print(sm)
        return sm


class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list,
                           starting_point=self.starting_point)
        self.seen_states = dict()
        self.targets = get_corners(self)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        upper_left = state.state.item((0, self.board_h-1))
        upper_right = state.state.item((self.board_w-1, self.board_h-1))
        down_right = state.state.item((self.board_w-1, 0))
        condition = upper_left != -1 and upper_right != -1 and down_right != -1
        return condition

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for
                move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves
        """
        new_board = Board(self.board_w, self.board_h, 1, self.piece_list,
                          starting_point=self.starting_point)
        for action in actions:
            new_board.do_move(0, action)

        return np.sum((new_board.state != -1).astype(np.int))


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.
    First, try to come up with an admissible heuristic; almost all admissible
    heuristics will be consistent as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!
    On the other hand, inadmissible or inconsistent heuristics may find
    optimal solutions, so be careful.
    """
    return get_heuristic_cost(state, problem)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list,
                           starting_point=self.starting_point)
        self.seen_states = dict()

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        done = True
        for target in self.targets:
            done &= (state.state.item(target) != -1)
        return done

    def get_successors(self, state):
        """
        state: Search state

        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for
                move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must
        be composed of legal moves
        """
        new_board = Board(self.board_w, self.board_h, 1, self.piece_list,
                          starting_point=self.starting_point)
        for action in actions:
            new_board.do_move(0, action)

        return np.sum((new_board.state != -1).astype(np.int))


def blokus_cover_heuristic(state, problem):
    return get_heuristic_cost(state, problem)


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=(0, 0)):
        self.expanded = 0
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        """
        This method should return a sequence of actions that covers all target
        locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a
        time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered
            target location
            add actions to backtrace

        return backtrace
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()



class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=(0, 0)):
        self.targets = targets.copy()
        "*** YOUR CODE HERE ***"

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def solve(self):
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()
