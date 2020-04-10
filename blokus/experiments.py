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