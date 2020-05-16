import numpy as np
import abc
import util
from game import Agent, Action


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

        successor_game_state = current_game_state.generate_successor(action=action)
        scores = []

        for possible_action in successor_game_state.get_legal_actions(0):
            state = successor_game_state.generate_successor(0, possible_action)
            scores.append(state.score)

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
    def minimax_algorithm(depth, cur_state, max_turn, base_score_function, target_depth, actions):
        if depth == target_depth:
            return base_score_function(cur_state), actions

        if max_turn:
            possible_states = []
            for possible_action in cur_state.get_legal_actions(0):
                state = cur_state.generate_successor(0, possible_action)
                next_actions = list(actions)
                next_actions.append(possible_action)
                possible_states.append(minimax_algorithm(depth + 1, state, False,
                                                        base_score_function, target_depth, next_actions))
            if len(possible_states) == 0:
                return 0, [Action.STOP]
            return max(possible_states, key=lambda x: x[0])

        else:
            possible_states = []
            for possible_action in cur_state.get_legal_actions(1):
                state = cur_state.generate_successor(1, possible_action)
                next_actions = list(actions)
                next_actions.append(possible_action)
                possible_states.append(minimax_algorithm(depth + 1, state, True,
                                                        base_score_function, target_depth, next_actions))
            if len(possible_states) == 0:
                return 0, [Action.STOP]
            return min(possible_states, key=lambda x: x[0])

    return minimax_algorithm(currentDepth, currentState, True, base_score_function,
                             currentDepth + targetDepthToAdd * 2, [])[1][0]


def alpha_beta_pruning(currentDepth, currentState, base_score_function, targetDepthToAdd):
    def alpha_beta_pruning_algorithm(depth, cur_state, max_turn, base_score_function, target_depth, alpha, beta,
                                     actions):
        if depth == target_depth:
            return base_score_function(cur_state), actions

        if max_turn:
            possible_states = []
            max_eval = -np.inf, [Action.STOP]
            for possible_action in cur_state.get_legal_actions(0):
                state = cur_state.generate_successor(0, possible_action)
                next_actions = list(actions)
                next_actions.append(possible_action)
                curr_eval = alpha_beta_pruning_algorithm(depth + 1, state, False, base_score_function, target_depth,
                                                         alpha, beta, next_actions)
                possible_states.append(curr_eval)
                max_eval = max(max_eval, curr_eval, key=lambda x: x[0])
                alpha = max(alpha, curr_eval[0])
                if beta <= alpha:
                    break
            if len(possible_states) == 0:
                return 0, [Action.STOP]
            return max(possible_states, key=lambda x: x[0])

        else:
            possible_states = []
            min_eval = np.inf, [Action.STOP]
            for possible_action in cur_state.get_legal_actions(1):
                state = cur_state.generate_successor(1, possible_action)
                next_actions = list(actions)
                next_actions.append(possible_action)
                curr_eval = alpha_beta_pruning_algorithm(depth + 1, state, True, base_score_function, target_depth,
                                                         alpha, beta, next_actions)
                possible_states.append(curr_eval)
                min_eval = min(min_eval, curr_eval, key=lambda x: x[0])
                beta = min(beta, curr_eval[0])
                if beta <= alpha:
                    break
            if len(possible_states) == 0:
                return 0, [Action.STOP]
            return min(possible_states, key=lambda x: x[0])

    return alpha_beta_pruning_algorithm(currentDepth, currentState, True, base_score_function,
                                        currentDepth + targetDepthToAdd * 2, -np.inf, np.inf, [])[1][0]


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
        return minimax(0, game_state, score_evaluation_function, self.depth)



class AlphaBetaAgent(MultiAgentSearchAgent):
    """
    Your minimax agent with alpha-beta pruning (question 3)
    """

    def get_action(self, game_state):
        """
        Returns the minimax action using self.depth and self.evaluationFunction
        """
        return alpha_beta_pruning(0, game_state, score_evaluation_function, self.depth)



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
        """*** YOUR CODE HERE ***"""
        util.raiseNotDefined()





def better_evaluation_function(current_game_state):
    """
    Your extreme 2048 evaluation function (question 5).

    DESCRIPTION: <write something here so we know what you did>
    """
    "*** YOUR CODE HERE ***"
    util.raiseNotDefined()


# Abbreviation
better = better_evaluation_function
