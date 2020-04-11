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


def get_played_positions(board):
    """
    Gets the positions that are played in the given board
    :param board: The current board at play
    :return: A numpy array in which every sub-array contains two integers,
    one for each coordinate of a position
    """
    return np.argwhere(board.state != -1)


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


def flip_board(board):
    return np.where(board.state == 0, 1, 0)


def get_complete_generalized_targets(generalized_targets, problem):
    to_return = generalized_targets
    tar_dict = dict()
    for tar in problem.targets:
        tar_dict[tuple(tar)] = False
    for gen_tar in generalized_targets:
        for tar in gen_tar.original:
            tar_dict[tuple(tar)] = True
    for tar in tar_dict:
        if not tar_dict[tuple(tar)]:
            to_return.append(GeneralizedTarget([tar], np.array([tar])))
    return to_return


def get_distance_between_components(generalized_targets):
    distances_dict = dict()
    for get_tar in generalized_targets:
        distances_dict[get_tar] = np.inf
    gen_tar_amount = len(generalized_targets)
    for i in range(0, gen_tar_amount):
        cur_gen_tar1 = generalized_targets[i]
        cur_gen_tar1_targets = cur_gen_tar1.targets
        for j in range(0, gen_tar_amount):
            if j != i:
                cur_gen_tar2 = generalized_targets[j]
                cur_gen_tar2_targets = cur_gen_tar2.targets
                print('cur_gen_tar1')
                print(cur_gen_tar1.targets)
                print('cur_gen_tar2')
                print(cur_gen_tar2.targets)
                distances_matrix = distance.cdist(cur_gen_tar1_targets,
                                                   cur_gen_tar2_targets,
                                                  'chebyshev')
                print('distances_matrix')
                print(distances_matrix)
                new_dist = np.min(distances_matrix)
                if distances_dict[cur_gen_tar1] > new_dist:
                    distances_dict[cur_gen_tar1] = new_dist
                if distances_dict[cur_gen_tar2] > new_dist:
                    distances_dict[cur_gen_tar2] = new_dist

    to_return = max(distances_dict.values())
    return to_return


def get_original_targets(from_labeled, problem):
    to_return = []
    lst = from_labeled.tolist()
    for index in lst:
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
    print('board.state')
    print(board.state)
    for i in range(1, components_num + 1):
        labeled_indices = indices[labeled == i]
        generalized_targets.append(GeneralizedTarget(get_original_targets(
            labeled_indices, problem), labeled_indices))
    to_return = get_distance_between_components(
        get_complete_generalized_targets(generalized_targets, problem))
    print('heuristic cost')
    print('heuristic cost')
    print('heuristic cost')
    print('heuristic cost')
    print('heuristic cost')
    print(to_return)
    print(to_return)
    print(to_return)
    print(to_return)
    print(to_return)
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
        original_targets = [[0, 0], [0, board.board_w - 1],
                      [board.board_h - 1, 0],
                      [board.board_h - 1, board.board_w - 1]]
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
            if is_state_authorized(new_board, self):
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
        total_cost = 0
        for action in actions:
            cost = action.piece.get_num_tiles()
            total_cost += cost
        return total_cost


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
        total_cost = 0
        for action in actions:
            cost = action.piece.get_num_tiles()
            total_cost += cost
        return total_cost


def blokus_cover_heuristic(board, problem):
    return


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
