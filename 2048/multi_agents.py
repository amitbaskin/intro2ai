import numpy as np
import abc
import util
from game import Agent, Action
import math


class ReflexAgent(Agent):
    """
    A reflex agent chooses an action at each choice point by examining
    its alternatives via a state evaluation function.

    The code below is provided as a guide.  You are welcome to change
    it in any way you see fit, so long as you don't touch our method
    headers.
    """

    def get_action(self, game_state):
        """
        You do not need to change this method, but you're welcome to.

        get_action chooses among the best options according to the evaluation function.

        get_action takes a game_state and returns some Action.X for some X in the set {UP, DOWN, LEFT, RIGHT, STOP}
        """

        # Collect legal moves and successor states
        legal_moves = game_state.get_agent_legal_actions()

        # Choose one of the best actions
        scores = [self.evaluation_function(game_state, action) for action in legal_moves]
        best_score = max(scores)
        best_indices = [index for index in range(len(scores)) if scores[index] == best_score]
        chosen_index = np.random.choice(best_indices)  # Pick randomly among the best

        "Add more of your code here if you want to"

        return legal_moves[chosen_index]

    def evaluation_function(self, current_game_state, action):
        """
        Design a better evaluation function here.

        The evaluation function takes in the current and proposed successor
        GameStates (GameState.py) and returns a number, where higher numbers are better.

        """

        # Useful information you can extract from a GameState (game_state.py)
        # Note that this decides the score by using 2 player 0 actions in a row. This may not make sense for the game,
        # but it's a heuristic.

        successor_game_state = current_game_state.generate_successor(action=action)
        scores = [successor_game_state.generate_successor(0, possible_action).score for possible_action in
                  successor_game_state.get_legal_actions(0)]

        return max(scores)


def score_evaluation_function(current_game_state):
    """
    This default evaluation function just returns the score of the state.
    The score is the same one displayed in the GUI.

    This evaluation function is meant for use with adversarial search agents
    (not reflex agents).
    """
    return current_game_state.score


class MultiAgentSearchAgent(Agent):
    """
    This class provides some common elements to all of your
    multi-agent searchers.  Any methods defined here will be available
    to the MinmaxAgent, AlphaBetaAgent & ExpectimaxAgent.

    You *do not* need to make any changes here, but you can if you want to
    add functionality to all your adversarial search agents.  Please do not
    remove anything, however.

    Note: this is an abstract class: one that should not be instantiated.  It's
    only partially specified, and designed to be extended.  Agent (game.py)
    is another abstract class.
    """

    def __init__(self, evaluation_function='scoreEvaluationFunction', depth=2):
        self.evaluation_function = util.lookup(evaluation_function, globals())
        self.depth = depth

    @abc.abstractmethod
    def get_action(self, game_state):
        return


def minimax(currentDepth, currentState, base_score_function, targetDepthToAdd):
    def minimax_algorithm(depth, cur_state, max_turn, base_score_function, target_depth, action):
        if depth == target_depth:
            return base_score_function(cur_state), action

        if max_turn:
            possible_states = []
            for possible_action in cur_state.get_legal_actions(0):
                state = cur_state.generate_successor(0, possible_action)
                if action is None:
                    curr_action = possible_action
                else:
                    curr_action = action
                possible_states.append(minimax_algorithm(depth + 1, state, False,
                                                        base_score_function, target_depth, curr_action))
            if len(possible_states) == 0:
                return -np.inf, action
            return max(possible_states, key=lambda x: x[0])

        else:
            possible_states = []
            for possible_action in cur_state.get_legal_actions(1):
                state = cur_state.generate_successor(1, possible_action)
                if action is None:
                    curr_action = possible_action
                else:
                    curr_action = action
                possible_states.append(minimax_algorithm(depth + 1, state, True,
                                                        base_score_function, target_depth, curr_action))
            if len(possible_states) == 0:
                return base_score_function(cur_state), action
            return min(possible_states, key=lambda x: x[0])

    return minimax_algorithm(currentDepth, currentState, True, base_score_function,
                             currentDepth + targetDepthToAdd * 2, None)[1]


def alpha_beta_pruning(currentDepth, currentState, base_score_function, targetDepthToAdd):
    def alpha_beta_pruning_algorithm(depth, cur_state, max_turn, base_score_function, target_depth, alpha, beta,
                                     action):
        if depth == target_depth:
            return base_score_function(cur_state), action

        if max_turn:
            possible_states = []
            max_eval = -np.inf, [Action.STOP]
            for possible_action in cur_state.get_legal_actions(0):
                state = cur_state.generate_successor(0, possible_action)
                if action is None:
                    curr_action = possible_action
                else:
                    curr_action = action
                curr_eval = alpha_beta_pruning_algorithm(depth + 1, state, False, base_score_function, target_depth,
                                                         alpha, beta, curr_action)
                possible_states.append(curr_eval)
                max_eval = max(max_eval, curr_eval, key=lambda x: x[0])
                alpha = max(alpha, curr_eval[0])
                if beta <= alpha:
                    break
            if len(possible_states) == 0:
                return base_score_function(cur_state), action
            return max(possible_states, key=lambda x: x[0])

        else:
            possible_states = []
            min_eval = np.inf, [Action.STOP]
            for possible_action in cur_state.get_legal_actions(1):
                state = cur_state.generate_successor(1, possible_action)
                if action is None:
                    curr_action = possible_action
                else:
                    curr_action = action
                curr_eval = alpha_beta_pruning_algorithm(depth + 1, state, True, base_score_function, target_depth,
                                                         alpha, beta, curr_action)
                possible_states.append(curr_eval)
                min_eval = min(min_eval, curr_eval, key=lambda x: x[0])
                beta = min(beta, curr_eval[0])
                if beta <= alpha:
                    break
            if len(possible_states) == 0:
                return base_score_function(cur_state), action
            return min(possible_states, key=lambda x: x[0])

    return alpha_beta_pruning_algorithm(currentDepth, currentState, True, base_score_function,
                                        currentDepth + targetDepthToAdd * 2, -np.inf, np.inf, None)[1]


def expectimax(currentDepth, currentState, base_score_function, targetDepthToAdd):
    def expectimax_algorithm(depth, cur_state, max_turn, base_score_function, target_depth, action):
        if depth == target_depth:
            return base_score_function(cur_state), action

        if max_turn:
            possible_states = []
            for possible_action in cur_state.get_agent_legal_actions():
                state = cur_state.generate_successor(0, possible_action)
                if depth == 0:
                    curr_action = possible_action
                else:
                    curr_action = action
                possible_states.append(expectimax_algorithm(depth + 1, state, False,
                                                            base_score_function, target_depth, curr_action))
            if len(possible_states) == 0:
                return base_score_function(cur_state), action
            return max(possible_states, key=lambda x: x[0])

        else:
            possible_states = []
            for possible_action in cur_state.get_opponent_legal_actions():
                state = cur_state.generate_successor(1, possible_action)
                if depth == 0:
                    curr_action = possible_action
                else:
                    curr_action = action
                possible_states.append(expectimax_algorithm(depth + 1, state, True,
                                                            base_score_function, target_depth, curr_action))
            if len(possible_states) == 0:
                return base_score_function(cur_state), action
            mean = sum([score for score, _ in possible_states]) / len(possible_states)
            return mean, action

    return expectimax_algorithm(currentDepth, currentState, True, base_score_function,
                                currentDepth + targetDepthToAdd * 2, None)[1]


def not_all_same(states):
    curr_value = states[0][0]
    for state in states:
        if state[0] != curr_value:
            return True
    return False


class MinmaxAgent(MultiAgentSearchAgent):
    def get_action(self, game_state):
        """
        Returns the minimax action from the current gameState using self.depth
        and self.evaluationFunction.

        Here are some method calls that might be useful when implementing minimax.

        game_state.get_legal_actions(agent_index):
            Returns a list of legal actions for an agent
            agent_index=0 means our agent, the opponent is agent_index=1

        Action.STOP:
            The stop direction, which is always legal

        game_state.generate_successor(agent_index, action):
            Returns the successor game state after an agent takes an action
        """
        return minimax(0, game_state, self.evaluation_function, self.depth)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return alpha_beta_pruning(0, game_state, self.evaluation_function, self.depth)



class ExpectimaxAgent(MultiAgentSearchAgent):
    """
    Your expectimax agent (question 4)
    """

    def get_action(self, game_state):
        """
        Returns the expectimax action using self.depth and self.evaluationFunction

        The opponent should be modeled as choosing uniformly at random from their
        legal moves.
        """
        return expectimax(0, game_state, self.evaluation_function, self.depth)


def board_monotonicity(board):
    score = 0

    score += np.sum(np.all((board[1:] >= board[:-1]) | (board[1:] == 0), axis=0))
    score += np.sum(np.all((board[:, 1:] >= board[:, :-1]) | (board[:, 1:] == 0), axis=1))

    score += np.sum(np.all((board[1:] <= board[:-1]) | (board[:-1] == 0), axis=0))
    score += np.sum(np.all((board[:, 1:] <= board[:, :-1]) | (board[:, :-1] == 0), axis=1))

    return score


def board_smoothness(board):
    score = 0

    for i in range(board.shape[0]):
        prev_cell = 0
        for j in range(board.shape[1]):
            if (abs(board[i, j] - prev_cell) > 0):
                if prev_cell != 0 and board[i, j] != 0:
                    score += abs(board[i, j] - prev_cell)
            prev_cell = board[i, j]

    board = board.T
    for i in range(board.shape[0]):
        prev_cell = 0
        for j in range(board.shape[1]):
            if (abs(board[i, j] - prev_cell) > 0):
                if prev_cell != 0 and board[i, j] != 0:
                    score += abs(board[i, j] - prev_cell)
            prev_cell = board[i, j]

    return -score


def biggest_num_in_corner(board):
    highest_tile_locations = np.argwhere(board == np.amax(board))
    h, w = board.shape
    for location in highest_tile_locations:
        if np.all(location == [0, 0]) or np.all(location == [0, w - 1]) or \
           np.all(location == [h - 1, 0]) or np.all(location == [h - 1, w - 1]):
            return board.max()
    return 0


def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION:
    To make sure that the board stays in a state that is conducive to merges we made a few heuristics:
    1. The board is monotone in a direction, this means that the highest tile is more likely to stay at a corner and the
       further away you get from it, the smaller the numbers get, this means that merges of small numbers happen in one
       corner of the board, and big merges happen in the opposite side. We ignore empty tiles that are in between the
       tiles.
    2. To make sure tiles stay in places where the can be merged, we want adjacent tiles to be similar. We ignore empty
       tiles that are in between the tiles.
    3. We want to make sure that there are places in which tiles can move, so we want to maximize the number of empty
       tiles.
    4. We obviously want the game to have the highest tile possible.
    5. Also for the sum  of the tiles on board.
    6. We want to make extra sure that the biggest tile is in one of the corners to make sure that this event cannot
       happen: The biggest tile moves for "one turn" to make something happen, and then a tile spawns in the corner
               where it would have returned to.
    7. Obviously, the score itself is important as well.

    All these heuristics are weighed in a manner such that they are all of the same order of magnitude more or less.
    """
    monotonicity = board_monotonicity(current_game_state.board) * 300
    smoothness = board_smoothness(current_game_state.board) * 10
    empty_tiles = np.sum(current_game_state.board == 0) * 200
    max_tile = current_game_state.max_tile * 10
    sum_tiles = np.sum(current_game_state.board) * 15
    in_corner = biggest_num_in_corner(current_game_state.board) * 100
    score = current_game_state.score

    return monotonicity + score + smoothness + empty_tiles + max_tile + sum_tiles + in_corner
    

# Abbreviation
better = better_evaluation_function
