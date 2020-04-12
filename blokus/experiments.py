from board import *

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
