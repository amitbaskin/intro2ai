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


def get_target_corners(problem):
    target_corners = [(0, 0), (0, problem.board_w - 1),
                      (problem.board_h - 1, 0),
                      (problem.board_h - 1, problem.board_w - 1)]
    if problem.starting_point in target_corners:
        target_corners.remove(problem.starting_point)
    return target_corners


def get_forbidden_adjacent_positions(problem, pos):
    adjacent_positions = [(pos[0] - 1, pos[1]), (pos[0] + 1, pos[1]),
                          (pos[0], pos[1] + 1), (pos[0], pos[1] - 1)]
    forbidden_adjacent_positions = []
    for pos in adjacent_positions:
        if problem.board.check_tile_legal(0, pos[0], pos[1]):
            forbidden_adjacent_positions.append(pos)
    return forbidden_adjacent_positions


def get_total_forbidden_positions(state, problem):
    forbidden_positions = []
    for target in get_current_targets(state, problem):
        forbidden_positions += get_forbidden_adjacent_positions(problem, target)
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
    current_targets = []
    for target in problem.targets:
        if state.state.item(target) == -1:
            current_targets.append(target)
    return current_targets


def get_min_dist_from_target(state, problem, target):
    played_positions = []
    for row_num in range(problem.board_h):
        for column_num in range(problem.board_w):
            cur_pos = (row_num, column_num)
            if state.state.item(cur_pos) != -1:
                played_positions.append((row_num, column_num))
    min_dist = np.inf
    for pos in played_positions:
        cur_dist = get_distance_between_positions(pos, target)
        if cur_dist < min_dist:
            min_dist = cur_dist
    return min_dist


def get_sum_distances_from_targets(state, problem):
    sm = 0
    cur_targets = get_current_targets(state, problem)
    len_cur_goals = len(cur_targets)
    if len_cur_goals == 0:
        return 0
    else:
        for corner in cur_targets:
            sm += get_min_dist_from_target(state, problem, corner)
        return sm


def get_cost_of_actions_helper(actions, board_w, board_h, piece_list,
                               starting_point):
    """
    actions: A list of actions to take

    This method returns the total cost of a particular sequence of actions.
    The sequence must
    be composed of legal moves
    """
    new_board = Board(board_w, board_h, 1, piece_list,
                      starting_point)
    for action in actions:
        new_board.do_move(0, action)
    return np.sum((new_board.state != -1).astype(np.int))


def get_diagonal_positions(pos, board):
    x = pos[0]
    y = pos[1]
    diag_pos = [(x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1), (x - 1, y - 1)]
    return [pos for pos in diag_pos if board.check_tile_legal(0, pos)]


def get_played_positions(board):
    return np.argwhere(board.state != -1)


def get_legal_diagonal_positions(board):
    played_positions = get_played_positions(board)
    diagonal_positions = {}
    for pos in played_positions:
        pos_tup = (pos[0], pos[1])
        diagonal_positions += get_diagonal_positions(pos_tup, board)
    return diagonal_positions


def get_updated_board(board_w, board_h, piece_list, board, position):
    board_state = board.state
    new_board = Board(board_w, board_h, 1, piece_list, position)
    new_board.state = board_state
    return new_board


def get_position_moves(board_w, board_h, piece_list, target):
    board = Board(board_w, board_h, 1, piece_list, target)
    return board.get_legal_moves(0)


def get_board_filled_targets(board_w, board_h, piece_list, board,
                             targets_moves):
    cur_board = board
    for target_move in targets_moves:
        cur_board = get_updated_board(board_w, board_h, piece_list, cur_board,
                                      target_move[0])
        cur_board.do_move(0, target_move[1])
    return cur_board


def generate_target_set(board, board_w, board_h, piece_list, target, move):
    new_board = get_updated_board(board_w, board_h, piece_list,
                                  board, target)
    new_board.do_move(0, move)
    return get_legal_diagonal_positions(board)


def get_targets_sets(board, board_w, board_h, piece_list, targets_moves):
    targets_sets = []
    for target_move in targets_moves:
        targets_sets.append(
            TargetSet(board, board_w, board_h, piece_list, target_move[0],
                      target_move[1]))
    return targets_sets


class TargetSet:
    def __init__(self, board, board_w, board_h, piece_list, target, move):
        self.target_set = generate_target_set(board, board_w, board_h,
                                              piece_list, target, move)

    def get_distance_from_target_set(self, board, problem):
        distances = []
        for target in self.target_set:
            distances.append(get_min_dist_from_target(board, problem, target))
        return min(distances)


class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, self.starting_point)
        self.targets = get_target_corners(self)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, state):
        if len(get_current_targets(state, self)) == 0:
            return True
        else:
            return False

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
        return get_cost_of_actions_helper(actions, self.board_w,
                                          self.board_h, self.piece_list,
                                          self.starting_point)


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
        self.board = Board(board_w, board_h, 1, piece_list, self.starting_point)

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
        return get_cost_of_actions_helper(actions, self.board_w,
                                          self.board_h, self.piece_list,
                                          self.starting_point)


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
