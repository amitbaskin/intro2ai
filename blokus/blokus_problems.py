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


class OriginalTargets:
    """
    The original targets given in a problem such that we can rotate along
     the targets get the next target in the list.
    """

    def __init__(self, original_targets, current_target_index=0):
        self.original_targets = original_targets
        self.current_target_index = current_target_index
        self.current_target = original_targets[current_target_index]
        self.targets_amount = len(original_targets)

    def next(self):
        """
        Gets the next target in the list of original_targets.
        :return: The next target in the list of original_targets
        """
        next_index = self.current_target_index + 1
        self.current_target_index = next_index % self.targets_amount
        return self.original_targets[self.current_target_index]


class GeneralizedTarget:
    """
    A generalization of a target. Instead of one position, here we have a
    list of positions which constitute the generalized target.
    """

    def __init__(self, target):
        self.targets = [target]
        self.original_target = target

    def get_distance(self, board, positions):
        """
        Gets the minimal distance between a position in the given positions
        and a position in *this* generalized target.
        :param board: A board which the only played positions in it are the
        given ones
        :param positions: The positions to measure the distance from *this*
        generalized target
        :return: The minimal distance between a position in the given positions
        and a position in *this* generalized target
        """
        new_board = filled_with_played_positions(positions, board)
        distances = []
        for target in self.targets:
            distances.append(get_min_dist_from_target(new_board, target))
        return min(distances)

    def update_targets(self, new_targets):
        """
        whenever we have new positions to attach to a given generalized
        target, we add the attached positions to the generalized target.
        :param new_targets: The new attached positions
        :return: null
        """
        self.targets += new_targets


class GeneralizedBoard:
    """
    A board which holds the information of its current generalized targets
    """

    def __init__(self, board, problem, starting_point, generalized_targets):
        self.board = board
        self.generalized_targets = generalized_targets
        self.move = None
        self.cost = 0
        self.heuristic_cost = 1
        self.cost_so_far = 0
        self.a_star_cost = 0


def initialize_generalized_targets(board, problem, starting_point):
    """
    Using the original targets of a problem, we generate generalized targets in
    which every generalized target contains only one original target or the
    starting point of the problem.
    :param board: The board in play
    :param problem: The problem we want to solve
    :param starting_point: The point the problem starts from
    :return: A list of generalized targets
    """
    generalized_targets = []
    original_targets = get_target_corners(board, problem) + [starting_point]
    for target in original_targets:
        generalized_targets.append(GeneralizedTarget(target))
    return generalized_targets


def get_generalized_targets_distances(generalized_targets, board):
    """
    Gets the total sum of distances between the generalized targets given.
    :param generalized_targets:
    :param board: The board in play
    :return: The sum
    """
    representative = generalized_targets[0]
    sm = 0
    for i in range(1, len(generalized_targets)):
        cur_gen_target = generalized_targets[i]
        sm += representative.get_distance(board, cur_gen_target)
    return sm


def get_move_positions(move):
    """
    Gets the positions which a the given move covers.
    :param move: The move to get its covered positions
    :return: The covered positions of the given move
    """
    move_positions = []
    for (xi, yi) in move.orientation:
        (x, y) = (xi + move.x, yi + move.y)
        move_positions.append((y, x))
    return move_positions


def attach_move_to_generalized_targets(move, generalized_board):
    """
    Adds the covered positions of the given move to the generalized targets
    which it is attached to.
    :param move: The given move to add its covered positions to the
    generalized targets which it is attached to
    :param problem: The problem of which we want to check its generalized
    targets whether the covered positions of the given moves are attached to
    :return: null
    """
    move_positions = get_move_positions(move)
    for generalized_target in generalized_board.generalized_targets:
        if generalized_target.get_distance(move_positions) == 0:
            generalized_target.update_targets(move_positions)


def filled_with_played_positions(played_positions, board):
    """
    Gets A new board in which the only played positions are the given played
    positions.
    :param played_positions: The positions to play
    :param board: The board in play
    :return: A new board in which the only played positions are the given played
    positions
    """
    new_board = board.__copy__()
    new_state = new_board.state
    for position, value in np.ndenumerate(board.state):
        if position in played_positions:
            new_board[position] = 0
        else:
            new_board[position] = -1
    new_board.state = new_state
    return new_board


def get_played_positions(board):
    """
    Gets the positions that are played in the given board
    :param board: The current board at play
    :return: A numpy array in which every sub-array contains two integers,
    one for each coordinate of a position
    """
    return np.argwhere(board.state != -1)


def get_diagonal_positions(pos, board):
    """
    Gets the diagonal positions from the given position, such that these
    positions are within the board and are not played.
    :param pos: The position of which we want to get its diagonal positions
    :param board: The board in play
    :return: A list of the diagonal positions
    """
    x = pos[0]
    y = pos[1]
    diag_pos = [(x + 1, y + 1), (x - 1, y + 1), (x + 1, y - 1), (x - 1, y - 1)]
    to_return = []
    for pos in diag_pos:
        if board.check_tile_legal(0, pos) and board.state.item(pos) == -1:
            to_return.append(pos)
    return to_return


def get_total_diagonal_positions(board):
    """
    Given a board, it gets all diagonal positions of currently played
    positions. The idea of this of function is to give it a board which the
    only played positions in it are the positions which we want to get the
    diagonal positions of, i.e. a certain target-set.
    :param board: A board in which the only played positions are those of a
    certain generalized target
    :return: A list of all diagonal positions from the played positions in
    the board
    """
    played_positions = get_played_positions(board)
    diagonal_positions = []
    for pos in played_positions:
        pos_tup = (pos[0], pos[1])
        diagonal_positions += get_diagonal_positions(pos_tup, board)
    return diagonal_positions


def updated_board(board_w, board_h, piece_list, board, position):
    """
    Gets a copy of the given board with updated starting position.
    :param board_w: The width of the board
    :param board_h: The height of the board
    :param piece_list: The current piece list
    :param board: The board in play
    :param position: The new starting position to be set
    :return: The new updated board
    """
    board_state = board.state
    new_board = Board(board_w, board_h, 1, piece_list, position)
    new_board.state = board_state
    return new_board


def get_target_corners(board, problem):
    """
    It is given to us in the corners problem that the starting point has to
    be some corner. So here we check which corner is the starting point and
    then give the other corners as our targets.
    :param board: The board in play
    :param problem: The problem we want to solve
    :return: The corners which are the targets of the corner problem
    """
    target_corners = [(0, 0), (0, board.board_w - 1),
                      (board.board_h - 1, 0),
                      (board.board_h - 1, board.board_w - 1)]
    if problem.starting_point in target_corners:
        target_corners.remove(board.starting_point)
    return target_corners


def get_forbidden_adjacent_positions(problem, target_position):
    """
    Given a target position which is played, we want to forbid playing the
    legal adjacent positions to it because if such position would be played
    then according to the rules of the Blokus game, it would be impossible to
    play the target position and so we would be in a losing state and waste
    our computation until we would realize that this is a losing state,
    so we want to prevent this the instant we can be aware of it which is
    when an adjacent position to a target position is played.
    :param problem: The corner problem we want to solve
    :param target_position: The target position we want to forbid playing
    the adjacent positions to it
    :return: The adjacent positions to the given target position, which are
    legal positions to play
    """
    adjacent_positions = [(target_position[0] - 1, target_position[1]),
                          (target_position[0] + 1, target_position[1]),
                          (target_position[0], target_position[1] + 1),
                          (target_position[0], target_position[1] - 1)]
    forbidden_adjacent_positions = []
    for target_position in adjacent_positions:
        if problem.board.check_tile_legal(0, target_position[0],
                                          target_position[1]):
            forbidden_adjacent_positions.append(target_position)
    return forbidden_adjacent_positions


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
    if is_state_authorized(board.state, problem):
        return get_sum_distances_from_targets(board.state, problem)
    else:
        return np.inf


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
        if board.state.item(target) == -1:
            current_targets.append(target)
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
    min_dist = np.inf
    for pos in played_positions:
        cur_dist = get_distance_between_positions(pos, target)
        if cur_dist < min_dist:
            min_dist = cur_dist
    return min_dist


def get_sum_distances_from_targets(board, problem):
    """
    Gets the sum of distances from the current targets.
    :param board: The board in play
    :param problem: The problem we want to solve
    :return: The sum of distances from the current targets
    """
    sm = 0
    cur_targets = get_current_targets(board, problem)
    len_cur_goals = len(cur_targets)
    if len_cur_goals == 0:
        return 0
    else:
        for target in cur_targets:
            sm += get_min_dist_from_target(board, target)
        return sm


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
    new_board = Board(board.board_w, board.board_h, 1, board.piece_list,
                      starting_point)
    for action in actions:
        new_board.do_move(0, action)
    return np.sum((new_board.state != -1).astype(np.int))


class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, starting_point)
        self.generalized_board = GeneralizedBoard(self.board,
                                                  self, starting_point)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, generalized_board):
        return get_generalized_targets_distances(
            self.generalized_board.generalized_targets, self.board) == 0

    def get_successors(self, board):
        """
        board: Search state
        For a given state, this should return a list of triples,
        (successor, action, stepCost), where 'successor' is a
        successor to the current state, 'action' is the action
        required to get there, and 'stepCost' is the incremental
        cost of expanding to that successor
        """
        # Note that for the search problem, there is only one player - #0
        self.expanded = self.expanded + 1
        return [(board.do_move(0, move), move, move.piece.get_num_tiles())
                for
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


class BlokusOriginalCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, self.starting_point)
        self.targets = get_target_corners(self.board, self)

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
    return get_heuristic_cost(board.state, problem)


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
