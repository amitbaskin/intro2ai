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

    def __init__(self, original, targets):
        self.original = original
        self.targets = targets

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

    def __init__(self, board, targets_amount):
        self.board = board
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
    :param generalized_board:
    :param move: The given move to add its covered positions to the
    generalized targets which it is attached to
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
        target_corners.remove(problem.starting_point)
    return target_corners


def get_forbidden_adjacent_positions(problem, pos):
    adjacent_positions = [[pos[0] - 1, pos[1]],
                          [pos[0] + 1, pos[1]],
                          [pos[0], pos[1] + 1],
                          [pos[0], pos[1] - 1]]

    to_return = []
    for pos in adjacent_positions:
        if problem.board.check_tile_legal(0, pos[0], pos[1]):
            to_return.append(pos)
    return np.array(to_return)


def get_all_illegal_positions(board, problem):
    played_positions = get_played_positions(board)
    to_return = played_positions.copy()
    for pos in played_positions:
        new_forbidden = get_forbidden_adjacent_positions(problem, pos)
        to_return = np.concatenate((to_return, new_forbidden))
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
        if board.state.item(target) == -1:
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
    new_board = Board(board.board_w, board.board_h, 1, board.piece_list,
                      starting_point)
    for action in actions:
        new_board.do_move(0, action)
    return np.sum((new_board.state != -1).astype(np.int))


def flip_board(board):
    return np.where(board.state == 0, 1, 0)


def reset_board(board):
    return np.where(board.state == 1, 0, -1)


def get_complete_generalized_targets(generalized_targets, problem):
    to_return = generalized_targets
    tar_dict = dict()
    for tar in problem.targets:
        tar_dict[tar] = False
    for gen_tar in generalized_targets:
        for tar in gen_tar.original:
            tar_dict[tar] = True
    for tar in tar_dict:
        if not tar_dict[tar]:
            to_return.append(GeneralizedTarget([tar], np.array([tar])))
    return to_return


def get_distance_between_components(generalized_targets):
    max_dist = 0
    gen_tar_amount = len(generalized_targets)
    for i in range(0, gen_tar_amount):
        cur_gen_tar1 = generalized_targets[i].targets
        for j in range(0, gen_tar_amount):
            if j != i:
                cur_gen_tar2 = generalized_targets[j].targets
                new_dist = np.min(
                    distance.cdist(cur_gen_tar1, cur_gen_tar2, 'chebyshev'))
                if max_dist < new_dist:
                    max_dist = new_dist
    return max_dist


def get_original_targets(labeled, problem):
    to_return = []
    for index, value in np.ndenumerate(labeled):
        if index in problem.targets:
            to_return.append(index)
    return to_return


def blokus_corners_heuristic(board, problem):
    state = flip_board(board)
    s = generate_binary_structure(2, 2)
    labeled, components_num = label(state, s)
    if components_num == 1:
        return 0
    indices = np.indices(state.shape).T[:, :, [1, 0]]
    generalized_targets = []
    for i in range(1, components_num + 1):
        labeled_indices = indices[labeled == i]
        generalized_targets.append(GeneralizedTarget(get_original_targets(
            labeled_indices, problem), labeled_indices))
    to_return = get_distance_between_components(
        get_complete_generalized_targets(generalized_targets, problem))
    return to_return


def update_legal_positions(board, problem):
    illegal_positions = get_all_illegal_positions(board, problem)
    array = board._legal
    array.fill(True)
    for pos in illegal_positions:
        array[0, pos[0], pos[1]] = False
    return array


def get_updated_board_starting_point(board, problem):
    starting_point = problem.original_targets.next()
    board.connected[0, starting_point[0], starting_point[1]] = True
    return board


def get_updated_board_all_starting_points(board, problem):
    for tar in problem.targets:
        board.connected[0, tar[0], tar[1]] = True
    return board


def get_updated_board_legal_positions(board, problem):
    board._legal = update_legal_positions(board, problem)
    return board


def fill_positions(board, positions):
    for pos in positions:
        board.state.itemset(pos, 0)


def check_move_validity(move, board):
    for (x, y) in move.orientation:
        # If any tile is illegal, this move isn't valid
        if not board.check_tile_legal(0, x + move.x, y + move.y):
            return False
    return True


def move_anyway(move, player, board):
    piece = move.piece
    board.pieces[player, move.piece_index] = False  # mark piece as used

    # Update internal state for each tile
    for (xi, yi) in move.orientation:
        (x, y) = (xi + move.x, yi + move.y)
        board.state[y, x] = player

        # This player can't play next to this square
        if x > 0:
            board._legal[player, y, x - 1] = False
        if x < board.board_w - 1:
            board._legal[player, y, x + 1] = False
        if y > 0:
            board._legal[player, y - 1, x] = False
        if y < board.board_h - 1:
            board._legal[player, y + 1, x] = False

        # The diagonals are now attached
        if x > 0 and y > 0:
            board.connected[player, y - 1, x - 1] = True
        if x > 0 and y < board.board_h - 1:
            board.connected[player, y + 1, x - 1] = True
        if x < board.board_w - 1 and y < board.board_h - 1:
            board.connected[player, y + 1, x + 1] = True
        if x < board.board_w - 1 and y > 0:
            board.connected[player, y - 1, x + 1] = True

    board.scores[player] += piece.get_num_tiles()
    return piece.get_num_tiles()


def is_targets_reached(board, problem):
    targets = problem.targets
    targets_amount = len(targets)
    counter = 0
    state = board.state
    for target in targets:
        if state[target[0], target[1]] == 0:
            counter += 1
    if counter == targets_amount:
        return True
    else:
        return False


class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        board = Board(board_w, board_h, 1, piece_list, starting_point)
        # fill_positions(self.board, original_targets)
        original_targets = get_target_corners(board, self) + [
            starting_point]
        self.original_targets = OriginalTargets(original_targets)
        self.targets = original_targets
        self.board = get_updated_board_all_starting_points(board, self)
        self.targets_amount = len(original_targets)

    def get_start_state(self):
        """
        Returns the start state for the search problem
        """
        return self.board

    def is_goal_state(self, board):
        print(board.state)
        if blokus_corners_heuristic(board, self) == 0 and \
                is_targets_reached(board, self):
            return True
        else:
            return False

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
        successors = []
        board = get_updated_board_starting_point(board, self)
        legal_moves = board.get_legal_moves(0)
        for move in legal_moves:
            new_board = board.do_move(0, move)
            newer_board = get_updated_board_legal_positions(new_board, self)
            cost = move.piece.get_num_tiles()
            successors.append((newer_board, move, cost))
        return successors

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must
        be composed of legal moves
        """
        return get_cost_of_actions_helper(actions, self.board,
                                          self.starting_point)


class OriginalBlokusCornersProblem(SearchProblem):
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


def original_blokus_corners_heuristic(board, problem):
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
