from board import *


class OriginalTargets:
    """
    The original targets given in a problem such that we can rotate along
     the targets get the next target in the list.
    """

    def __init__(self, positions, current_target_index=0):
        self.positions = positions
        self.current_target_index = current_target_index
        self.current_target = positions[current_target_index]
        self.targets_amount = len(positions)

    def next(self):
        """
        Gets the next target in the list of original_targets.
        :return: The next target in the list of original_targets
        """
        next_index = self.current_target_index + 1
        self.current_target_index = next_index % self.targets_amount
        updated_pos = self.positions[self.current_target_index]
        self.current_target = updated_pos
        return updated_pos


# def get_successors_helper(self, board, successors):
#     # board = get_updated_board_starting_point(board, self)
#     cur_targets_reached_amount = get_reached_targets_amount(board, self)
#     legal_moves = board.get_legal_moves(0)
#     for move in legal_moves:
#         new_board = board.do_move(0, move)
#         newer_board = get_updated_board_legal_positions(new_board, self)
#         new_targets_reached_amount = get_reached_targets_amount(
#             newer_board, self)
#         if is_targets_reached(board, self) and is_state_authorized(
#                 board, self):
#             cost = move.piece.get_num_tiles()
#             successors.append((newer_board, move, cost))
#             continue
#         if new_targets_reached_amount > cur_targets_reached_amount and \
#                 is_state_authorized(board,
#                                     self):
#             # print(new_board.state)
#             self.get_successors_helper(new_board, successors)
#     return successors


# def get_successors(self, board):
# #     """
# #     board: Search state
# #     For a given state, this should return a list of triples,
# #     (successor, action, stepCost), where 'successor' is a
# #     successor to the current state, 'action' is the action
# #     required to get there, and 'stepCost' is the incremental
# #     cost of expanding to that successor
# #     """
# #     # Note that for the search problem, there is only one player - #0
#     self.expanded = self.expanded + 1
#     successors = []
#     board = get_updated_board_starting_point(board, self)
#     legal_moves = board.get_legal_moves(0)
#     for move in legal_moves:
#         new_board = board.do_move(0, move)
#         if is_position_played(new_board,
#                               get_current_original_target(self)) and \
#                 is_state_authorized(
#                     board, self):
#             newer_board = get_updated_board_legal_positions(new_board, self)
#             cost = move.piece.get_num_tiles()
#             successors.append((newer_board, move, cost))
#     return successors


def get_updated_board_starting_point(board, problem):
    starting_point = problem.original_targets.next()
    board.connected[0, starting_point[0], starting_point[1]] = True
    return board


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


def get_played_positions(board):
    """
    Gets the positions that are played in the given board
    :param board: The current board at play
    :return: A numpy array in which every sub-array contains two integers,
    one for each coordinate of a position
    """
    return np.argwhere(board.state != -1)


def update_legal_positions(board, problem):
    illegal_positions = get_all_illegal_positions(board, problem)
    array = board._legal
    array.fill(True)
    for pos in illegal_positions:
        array[0, pos[0], pos[1]] = False
    return array


def get_updated_board_legal_positions(board, problem):
    board._legal = update_legal_positions(board, problem)
    return board


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


def get_min_tiles_remained(board):
    mn = np.inf
    pieces = board.piece_list
    for piece in pieces:
        cur_num = piece.get_num_tiles()
        if cur_num < mn:
            mn = cur_num
    return mn


def get_current_original_target(problem):
    return problem.original_targets.current_target


def is_position_played(board, pos):
    state = board.state
    pos_value = state[pos[0], pos[1]]
    return pos_value != -1


def connect_targets(board, problem):
    for tar in problem.targets:
        board.connected[0, tar[0], tar[1]] = True
    return board


def order_moves(moves, problem):
    board = Board(problem.board_w, problem.board_h, 1,
                  problem.piece_list, problem.starting_point)

    lst1 = []
    lst2 = moves
    while len(lst2) > 0:
        for move in lst2:
            if board.check_move_valid(0, move):
                board = board.do_move(0, move)
                lst1.append(move)
        lst2 = [item for item in lst2 if item not in lst1]
        # print(len(lst2))
    return lst1

# class GeneralizedTarget:
#     """
#     A generalization of a target. Instead of one position, here we have a
#     list of positions which constitute the generalized target.
#     """
#
#     def __init__(self, original, targets):
#         self.original = original
#         self.targets = targets
#
#     def get_distance(self, board, positions):
#         """
#         Gets the minimal distance between a position in the given positions
#         and a position in *this* generalized target.
#         :param board: A board which the only played positions in it are the
#         given ones
#         :param positions: The positions to measure the distance from *this*
#         generalized target
#         :return: The minimal distance between a position in the given positions
#         and a position in *this* generalized target
#         """
#         new_board = filled_with_played_positions(positions, board)
#         distances = []
#         for target in self.targets:
#             distances.append(get_min_dist_from_target(new_board, target))
#         return min(distances)
#
#     def update_targets(self, new_targets):
#         """
#         whenever we have new positions to attach to a given generalized
#         target, we add the attached positions to the generalized target.
#         :param new_targets: The new attached positions
#         :return: null
#         """
#         self.targets += new_targets


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


# def initialize_generalized_targets(board, problem, starting_point):
#     """
#     Using the original targets of a problem, we generate generalized targets in
#     which every generalized target contains only one original target or the
#     starting point of the problem.
#     :param board: The board in play
#     :param problem: The problem we want to solve
#     :param starting_point: The point the problem starts from
#     :return: A list of generalized targets
#     """
#     generalized_targets = []
#     original_targets = get_target_corners(board, problem) + [starting_point]
#     for target in original_targets:
#         generalized_targets.append(GeneralizedTarget(target))
#     return generalized_targets


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


# def get_total_diagonal_positions(board):
#     """
#     Given a board, it gets all diagonal positions of currently played
#     positions. The idea of this of function is to give it a board which the
#     only played positions in it are the positions which we want to get the
#     diagonal positions of, i.e. a certain target-set.
#     :param board: A board in which the only played positions are those of a
#     certain generalized target
#     :return: A list of all diagonal positions from the played positions in
#     the board
#     """
#     played_positions = get_played_positions(board)
#     diagonal_positions = []
#     for pos in played_positions:
#         pos_tup = (pos[0], pos[1])
#         diagonal_positions += get_diagonal_positions(pos_tup, board)
#     return diagonal_positions


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


def reset_board(board):
    return np.where(board.state == 1, 0, -1)


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


def fill_positions(board, positions):
    for pos in positions:
        board.state.itemset(pos, 0)


def check_move_validity(move, board):
    for (x, y) in move.orientation:
        # If any tile is illegal, this move isn't valid
        if not board.check_tile_legal(0, x + move.x, y + move.y):
            return False
    return True
