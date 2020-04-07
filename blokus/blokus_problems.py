from board import Board
from search import SearchProblem, ucs
import util
import numpy as np


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
        return [(state.do_move(0, move), move, 1) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        return len(actions)



#####################################################
# This portion is incomplete.  Time to write code!  #
#####################################################
class BlokusCornersProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0)):
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, starting_point=self.starting_point)
        self.seen_states = dict()

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
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        new_board = Board(self.board_w, self.board_h, 1, self.piece_list, starting_point=self.starting_point)
        for action in actions:
            new_board.do_move(0, action)

        return np.sum((new_board.state != -1).astype(np.int))


def euclidean_distance(point1, point2):
    # return np.floor(np.abs(np.array(point1) - np.array(point2)).sum() / 2)
    # return np.linalg.norm(np.array(point1) - np.array(point2)) / np.sqrt(2)
    return np.linalg.norm(np.array(point1) - np.array(point2))


def blokus_corners_heuristic(state, problem):
    """
    Your heuristic for the BlokusCornersProblem goes here.

    This heuristic must be consistent to ensure correctness.  First, try to come up
    with an admissible heuristic; almost all admissible heuristics will be consistent
    as well.

    If using A* ever finds a solution that is worse uniform cost search finds,
    your heuristic is *not* consistent, and probably not admissible!  On the other hand,
    inadmissible or inconsistent heuristics may find optimal solutions, so be careful.
    """
    if state in problem.seen_states:
        return problem.seen_states[state]

    goal_corners = []
    if state.state[state.board_h - 1, 0] == -1:
        goal_corners.append((state.board_h - 1, 0))
    if state.state[0, state.board_w - 1] == -1:
        goal_corners.append((0, state.board_w - 1))
    if state.state[state.board_h - 1, state.board_w - 1] == -1:
        goal_corners.append((state.board_h - 1, state.board_w - 1))
    if goal_corners.count == 0:
        return 0

    legal_corners = set()

    if np.all(state.state == -1):
        legal_corners.add((0, 0))
    else:
        for y in range(state.board_h):
            for x in range(state.board_w):
                if state.check_tile_attached(0, x, y) and state.check_tile_legal(0, x, y):
                    legal_corners.add((y, x))

    min_dist = np.inf
    #
    # closest_point_pairs = dict()
    # closest_point_dist = dict()
    # for goal_corner in goal_corners:
    #     closest_point_pairs[goal_corner] = None
    #     closest_point_dist[goal_corner] = np.inf
    #
    # for legal_corner in legal_corners:
    #     for goal_corner in goal_corners:
    #         dist = euclidean_distance(np.array(legal_corner), np.array(goal_corner))
    #         if dist < closest_point_dist[goal_corner]:
    #             closest_point_dist[goal_corner] = dist
    #             closest_point_pairs[goal_corner] = legal_corner
    #
    # dist = 0
    # for min_dist in closest_point_dist.values():
    #     dist += min_dist

    for legal_corner in legal_corners:
        # max_dist = 0
        # farthest_corner = None
        # for goal_corner in goal_corners:
        #     dist = euclidean_distance(legal_corner, np.array(goal_corner))
        #     if dist > max_dist:
        #         max_dist = dist
        #         farthest_corner = np.array(goal_corner)
        #
        # goal_corners_trimmed = []
        # for goal_corner in goal_corners:
        #     if not np.array_equal(goal_corner, farthest_corner):
        #         goal_corners_trimmed.append(goal_corner)
        #
        # curr_dist = max_dist
        # vec = farthest_corner - legal_corner
        # vec = vec / np.linalg.norm(vec, 2)
        # for goal_corner in goal_corners_trimmed:
        #     projection = np.floor(legal_corner + vec * np.dot(goal_corner - legal_corner, vec))
        #     # print(projection)
        #     curr_dist += euclidean_distance(projection, goal_corner)
        #
        # if max_dist < min_dist:
        #     min_dist = max_dist

        # all_dist = 0
        # for goal_corner in goal_corners:
        #     all_dist += np.round(euclidean_distance(goal_corner, legal_corner))
        # if all_dist < min_dist:
        #     min_dist = all_dist

        all_dist = 0
        for goal_corner in goal_corners:
            all_dist += util.manhattanDistance(goal_corner, legal_corner)
        if all_dist < min_dist:
            min_dist = all_dist

    print(min_dist / len(goal_corners))
    problem.seen_states[state] = min_dist / len(goal_corners)
    return min_dist / len(goal_corners)


class BlokusCoverProblem(SearchProblem):
    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=[(0, 0)]):
        self.targets = targets.copy()
        self.expanded = 0
        self.board_w = board_w
        self.board_h = board_h
        self.piece_list = piece_list
        self.starting_point = starting_point
        self.board = Board(board_w, board_h, 1, piece_list, starting_point=self.starting_point)
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
        return [(state.do_move(0, move), move, move.piece.get_num_tiles()) for move in state.get_legal_moves(0)]

    def get_cost_of_actions(self, actions):
        """
        actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.  The sequence must
        be composed of legal moves
        """
        new_board = Board(self.board_w, self.board_h, 1, self.piece_list, starting_point=self.starting_point)
        for action in actions:
            new_board.do_move(0, action)

        return np.sum((new_board.state != -1).astype(np.int))


def blokus_cover_heuristic(state, problem):
    if state in problem.seen_states:
        return problem.seen_states[state]

    goals = []
    for target in problem.targets:
        if state.state.item(target) == -1:
            goals.append(target)

    legal_corners = set()

    if np.all(state.state == -1):
        legal_corners.add((0, 0))
    else:
        for y in range(state.board_h):
            for x in range(state.board_w):
                if state.check_tile_attached(0, x, y) and state.check_tile_legal(0, x, y):
                    legal_corners.add((y, x))

    min_dist = np.inf

    for legal_corner in legal_corners:
        all_dist = 0
        for goal_corner in goals:
            all_dist += util.manhattanDistance(goal_corner, legal_corner)
        if all_dist < min_dist:
            min_dist = all_dist

    print(min_dist / len(goals))
    problem.seen_states[state] = min_dist
    return min_dist / len(goals)


class ClosestLocationSearch:
    """
    In this problem you have to cover all given positions on the board,
    but the objective is speed, not optimality.
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
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
        This method should return a sequence of actions that covers all target locations on the board.
        This time we trade optimality for speed.
        Therefore, your agent should try and cover one target location at a time. Each time, aiming for the closest uncovered location.
        You may define helpful functions as you wish.

        Probably a good way to start, would be something like this --

        current_state = self.board.__copy__()
        backtrace = []

        while ....

            actions = set of actions that covers the closets uncovered target location
            add actions to backtrace

        return backtrace
        """
        "*** YOUR CODE HERE ***"
        util.raiseNotDefined()



class MiniContestSearch:
    """
    Implement your contest entry here
    """

    def __init__(self, board_w, board_h, piece_list, starting_point=(0, 0), targets=(0, 0)):
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

