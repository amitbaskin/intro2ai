from board import Board
from search import SearchProblem, ucs
import util
import numpy as np
import math
from scipy.ndimage.measurements import label
from scipy.ndimage.morphology import generate_binary_structure
from scipy.spatial.distance import cdist
from scipy.spatial import distance


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

    def is_goal_state(self, board):
        """
        state: Search state
        Returns True if and only if the state is a valid goal state
        """
        return not any(board.pieces[0])

    def get_successors(self, board):
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
        return [(board.do_move(0, move), move, 1) for move in
                board.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves
        """
        return len(actions)


def get_target_corners(board, problem):
    """
    It is given to us in the corners problem that the starting point has to
    be some corner. So here we check which corner is the starting point and
    then give the other corners as our targets.
    :param board: The board in play
    :param problem: The problem we want to solve
    :return: The corners which are the targets of the corner problem
    """
    target_corners = [[0, 0], [0, board.board_w - 1],
                      [board.board_h - 1, 0],
                      [board.board_h - 1, board.board_w - 1]]
    if problem.starting_point in target_corners:
        target_corners.remove(problem.starting_point)
    return target_corners


def get_forbidden_adjacent_positions(problem, pos):
    adjacent_positions = [(pos[0] - 1, pos[1]),
                          (pos[0] + 1, pos[1]),
                          (pos[0], pos[1] + 1),
                          (pos[0], pos[1] - 1)]

    to_return = []
    for pos in adjacent_positions:
        if problem.board.check_tile_legal(0, pos[0], pos[1]):
            to_return.append(pos)
    return to_return


def get_total_forbidden_positions(board, problem):
    """
    For each target of the problem we want to get the forbidden positions
    of the target. So we get all the forbidden positions of the given problem.
    :param board: The board in play
    :param problem: The problem we want to solve
    :return: A list of all forbidden positions of the given problem
    """
    forbidden_positions = []
    for target in get_current_targets(board, problem):
        forbidden_positions += get_forbidden_adjacent_positions(problem, target)
    return forbidden_positions


def is_state_authorized(board, problem):
    """
    Determines whether or not a given board is allowed according to whether
    or not there are forbidden positions played in this board.
    :param board: The board to check its validation
    :param problem: The problem we want to solve
    :return: True iff the
    """
    forbidden_positions = get_total_forbidden_positions(board, problem)
    for pos in forbidden_positions:
        if board.state.item(pos) != -1:
            return False
    return True


def get_heuristic_cost(board, problem):
    """
    Gets the heuristic cost of given state.
    :param board: The board to check its state heuristic cost
    :param problem: The problem we want to solve
    :return: The heuristic cost of given state
    """
    # if is_state_authorized(board, problem):
    return get_max_distance_from_target(board, problem)
    # else:
    #     return np.inf


def get_distance_between_positions(pos1, pos2):
    """
    Gets the maximum difference between the coordinates of the two given
    positions.
    :param pos1: first position
    :param pos2: second position
    :return: The maximum difference between the coordinates of the two given
    positions
    """
    x_diff = math.fabs(pos1[0] - pos2[0])
    y_diff = math.fabs(pos1[1] - pos2[1])
    return max(x_diff, y_diff)


def get_current_targets(board, problem):
    """
    Gets the targets of the given problem which were not reached yet.
    :param board: The board in play
    :param problem: The problem we want to solve
    :return: The targets of the given problem which were not reached yet
    """
    current_targets = []
    for target in problem.targets:
        if board.state.item(tuple(target)) == -1:
            current_targets.append(target)
    problem.targets_amount = len(current_targets)
    return current_targets


def get_min_dist_from_target(board, target):
    """
    Gets the minimum distance between the played positions in the given
    board and the given target.
    :param board: The board in play
    :param target: The target to check the distance from
    :return: The minimum distance between the played positions in the given
     board and the given target.
    """
    played_positions = []
    for position, value in np.ndenumerate(board.state):
        if value != -1:
            played_positions.append(position)
    played_positions.append([0, 0])
    min_dist = np.inf
    for pos in played_positions:
        cur_dist = get_distance_between_positions(pos, target)
        if cur_dist < min_dist:
            min_dist = cur_dist
    return min_dist


def get_max_distance_from_target(board, problem):
    """
    Gets the sum of distances from the current targets.
    :param board: The board in play
    :param problem: The problem we want to solve
    :return: The sum of distances from the current targets
    """
    mx = 0
    cur_targets = get_current_targets(board, problem)
    len_cur_goals = len(cur_targets)
    if len_cur_goals == 0:
        return 0
    else:
        for target in cur_targets:
            current_dist = get_min_dist_from_target(board, target)
            if mx < current_dist:
                mx = current_dist
    print(board.state)
    print(mx)
    return mx


def get_cost_of_actions_helper(actions, board, starting_point):
    """
    Gets the total cost of a particular sequence of actions.
    The sequence must
    be composed of legal moves
    :param actions: A list of actions to take
    :param board: The board in play
    :param starting_point: The starting point in the board
    :return: The total cost of a particular sequence of actions.
    """
    total_cost = 0
    for action in actions:
        cost = action.piece.get_num_tiles()
        total_cost += cost
    return total_cost


class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, self.starting_point)
        self.targets = get_target_corners(self.board, self)
        self.targets_amount = len(self.targets)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, board):
        if len(get_current_targets(board, self)) == 0:
            return True
        else:
            return False

    def get_successors(self, board):
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
        successors = []
        for move in board.get_legal_moves(0):
            new_board = board.do_move(0, move)
            cost = move.piece.get_num_tiles()
            if is_state_authorized(new_board, self):
                successors.append((new_board, move, cost))
        return successors

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must
        be composed of legal moves
        """
        total_cost = 0
        for action in actions:
            cost = action.piece.get_num_tiles()
            total_cost += cost
        return total_cost


def blokus_corners_heuristic(board, problem):
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
    return get_heuristic_cost(board, problem)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0),
                 targets=[(0, 0)]):
        self.targets = targets
        self.targets_amount = len(targets)
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

    def is_goal_state(self, board):
        done = True
        for target in self.targets:
            done &= (board.state.item(target) != -1)
        return done

    def get_successors(self, board):
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
        return [(board.do_move(0, move), move, move.piece.get_num_tiles()) for
                move in board.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must
        be composed of legal moves
        """
        return get_cost_of_actions_helper(actions, self.board,
                                          self.starting_point)


def blokus_cover_heuristic(board, problem):
    return get_heuristic_cost(board, problem)


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
