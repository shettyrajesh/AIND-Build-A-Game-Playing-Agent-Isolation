"""This file contains all the classes you must complete for this project.

You can use the test cases in agent_test.py to help during development, and
augment the test suite with your own test cases to further test your code.

You must test your agent's strength against a set of agents with known
relative strength using tournament.py and include the results in your report.
"""
import random
import sys
import timeit

import isolation

class Timeout(Exception):
    """Subclass base exception for code clarity."""
    pass


def custom_score(game, player):
    """Calculate the heuristic value of a game state from the point of view
    of the given player.

    Note: this function should be called from within a Player instance as
    `self.score()` -- you should not need to call this function directly.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    -------
    float
        The heuristic value of the current game state to the specified player.
    """
    return(heuristic_D(game, player))


def heuristic_A(game, player):
    """
    Worst Performance of the lot.
    If the Student has fewer move options than the opponent, the function returns - (# moves the opponent has)/(# moves the computer has)**2.
    If the Student has more move options than the opponent, the function returns (# moves the opponent has)/(# moves the computer has)**2.
    The logic for this is similar. If a move causes the Student to have greatly proportionally fewer moves than the opponent, it is a bad move.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    own_moves = (game.get_legal_moves(player))
    opp_moves = (game.get_legal_moves(game.get_opponent(player)))

    if len (own_moves) == 0 and len (opp_moves) != 0:         #If this move results in the opponent winning the game, return utility -infinity
        return float ('-inf')
    elif len (own_moves) != 0 and len (opp_moves) == 0:       #If this move results in the computer winning the game, return utility infinity
        return float ('inf')
    elif len (own_moves) == 0 and len (opp_moves) == 0:       #If this move results in a draw, return utility -10
        return -10
    elif len (own_moves) >= len (opp_moves):
        return (len (own_moves) / len (opp_moves))**2
    elif len (own_moves) < len (opp_moves):
        return - (len (opp_moves)/ len(own_moves))**2


def heuristic_B(game, player):
    """
    Aggressive play in the first half of the game. Active player will try to choose the most aggressive move.
    Heuristic calculates number of players move vs 3.5 of value of an opponent’s moves.
    In the second half of the game heuristic will calculate number of players move vs number of an opponent’s moves.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    cells_left = game.width * game.height - game.move_count

    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if cells_left < int((game.width * game.height) / 2):
        return float(own_moves - opp_moves)
    return float(own_moves - 3 * opp_moves)


def heuristic_C(game, player):
    """Best Heuristic from the test results. Most aggressive initially, then drop aggressiveness at 1/3 and further more at 1/4 moves remaining.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    board_size = game.width * game.height
    cells_left = board_size - game.move_count
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if cells_left < int((board_size) / 4):
        return float(own_moves - opp_moves)
    if cells_left < int((board_size) / 3):
        return float(own_moves - 2 * opp_moves)
    return float(own_moves - 3 * opp_moves)


def heuristic_D(game, player):
    """Least aggressive initially, then increase aggressiveness at 1/3 and further more at 1/4 moves remaining.

    Parameters
    ----------
    game : `isolation.Board`
        An instance of `isolation.Board` encoding the current state of the
        game (e.g., player locations and blocked cells).

    player : object
        A player instance in the current game (i.e., an object corresponding to
        one of the player objects `game.__player_1__` or `game.__player_2__`.)

    Returns
    ----------
    float
    The heuristic value of the current game state to the specified player.
    """
    board_size = game.width * game.height
    cells_left = board_size - game.move_count
    if game.is_loser(player):
        return float("-inf")

    if game.is_winner(player):
        return float("inf")

    own_moves = len(game.get_legal_moves(player))
    opp_moves = len(game.get_legal_moves(game.get_opponent(player)))

    if cells_left < int((board_size) / 4):
        return float(own_moves - 3 * opp_moves)
    if cells_left < int((board_size) / 3):
        return float(own_moves - 2 * opp_moves)
    return float(own_moves - opp_moves)


class CustomPlayer:
    """Game-playing agent that chooses a move using your evaluation function
    and a depth-limited minimax algorithm with alpha-beta pruning. You must
    finish and test this player to make sure it properly uses minimax and
    alpha-beta to return a good move before the search time limit expires.

    Parameters
    ----------
    search_depth : int (optional)
        A strictly positive integer (i.e., 1, 2, 3,...) for the number of
        layers in the game tree to explore for fixed-depth search. (i.e., a
        depth of one (1) would only explore the immediate sucessors of the
        current state.)

    score_fn : callable (optional)
        A function to use for heuristic evaluation of game states.

    iterative : boolean (optional)
        Flag indicating whether to perform fixed-depth search (False) or
        iterative deepening search (True).

    method : {'minimax', 'alphabeta'} (optional)
        The name of the search method to use in get_move().

    timeout : float (optional)
        Time remaining (in milliseconds) when search is aborted. Should be a
        positive value large enough to allow the function to return before the
        timer expires.
    """

    def __init__(self, search_depth=3, score_fn=custom_score,
                 iterative=True, method='minimax', timeout=10.):
        self.search_depth = search_depth
        self.iterative = iterative
        self.score = score_fn
        self.method = method
        self.time_left = None
        self.TIMER_THRESHOLD = timeout


    def get_move(self, game, legal_moves, time_left):
        """Search for the best move from the available legal moves and return a
        result before the time limit expires.

        This function must perform iterative deepening if self.iterative=True,
        and it must use the search method (minimax or alphabeta) corresponding
        to the self.method value.

        **********************************************************************
        NOTE: If time_left < 0 when this function returns, the agent will
              forfeit the game due to timeout. You must return _before_ the
              timer reaches 0.
        **********************************************************************

        Parameters
        ----------
        game : `isolation.Board`
            An instance of `isolation.Board` encoding the current state of the
            game (e.g., player locations and blocked cells).

        legal_moves : list<(int, int)>
            A list containing legal moves. Moves are encoded as tuples of pairs
            of ints defining the next (row, col) for the agent to occupy.

        time_left : callable
            A function that returns the number of milliseconds left in the
            current turn. Returning with any less than 0 ms remaining forfeits
            the game.

        Returns
        -------
        (int, int)
            Board coordinates corresponding to a legal move; may return
            (-1, -1) if there are no available legal moves.
        """
        score_and_move_tuple = []
        self.time_left = time_left

        # TODO: finish this function!

        # Perform any required initializations, including selecting an initial
        # move from the game board (i.e., an opening book), or returning
        # immediately if there are no legal moves

        try:
            # The search method call (alpha beta or minimax) should happen in
            # here in order to avoid timeout. The try/except block will
            # automatically catch the exception raised by the search method
            # when the timer gets close to expiring
            move = []
            previous_best_result = []
            if (self.iterative != True):
                if (self.method == 'minimax'):
                    score_and_move_tuple.append(self.minimax(game, self.search_depth))
                elif (self.method == 'alphabeta'):
                    score_and_move_tuple.append(self.alphabeta(game, self.search_depth))
                return score_and_move_tuple[0][1]
            else: # (self.iterative = True)
                for depth in range(sys.maxsize ** 10):
                    if (self.method == 'minimax'):
                        _, move = self.minimax(game, depth+1)
                    elif (self.method == 'alphabeta'):
                        _, move = self.alphabeta(game, depth+1)
                    previous_best_result.append(move)
                if move != None:
                    return move

        except Timeout:
            # Handle any actions required at timeout, if necessary
            if (move != (-1, -1) and move != None):
                return move
            else:
                return max(previous_best_result)

        # Return the best move from the last completed search iteration


    def max_value_MM(self, board, minimax_call_depth, depth):
        '''
        Minimax Helper function to handle the maximizing player
        function MAX-VALUE(state) returns a utility value
            if TERMINAL-TEST(state) then
                return UTILITY(state)
            v ← −∞
            for each a in ACTIONS(state) do
                    v ← MAX(v, MIN-VALUE(RESULT(s, a)))
            return v
        '''
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        minimax_call_depth = minimax_call_depth + 1
        score_and_move_tuple = []

        if (board.is_winner(self) or board.is_loser(self) or (minimax_call_depth == depth) or (len(board.get_legal_moves()) == 0)):
            return self.score(board, self)

        for move in board.get_legal_moves():
            score_and_move_tuple.append(( self.min_value_MM(board.forecast_move(move), minimax_call_depth, depth)))
        return max(score_and_move_tuple)

    def min_value_MM(self, board, minimax_call_depth, depth):
        '''
        Minimax Helper function to handle the minimizing player

        function MIN-VALUE(state) returns a utility value
            if TERMINAL-TEST(state) then
                return UTILITY(state)
            v←∞
            for each a in ACTIONS(state) do
                v ← MIN(v, MAX-VALUE(RESULT(s, a)))
            return v
        '''

        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        minimax_call_depth = minimax_call_depth + 1
        score_and_move_tuple = []
        if (board.is_winner(self) or board.is_loser(self) or (minimax_call_depth == depth) or (len(board.get_legal_moves()) == 0)):
            return self.score(board, self)

        for move in board.get_legal_moves():
            score_and_move_tuple.append(( self.max_value_MM(board.forecast_move(move), minimax_call_depth, depth)))
        return min(score_and_move_tuple)


    def minimax(self, game, depth, maximizing_player=True):
        """Implement the minimax search algorithm as described in the lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        minimax_call_depth = 0
        score_and_move_tuple = []
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if(len(game.get_legal_moves()) == 0):
            return(self.score(game, self), (-1, -1))

        for move in game.get_legal_moves():
            next_state = game.forecast_move(move)
            score_and_move_tuple.append((self.min_value_MM(next_state, minimax_call_depth, depth), move))

        return max(score_and_move_tuple, key=lambda t: t[0])


    def max_value_AB(self, board, alpha, beta, minimax_call_depth, depth):
        '''
        Alphabeta Helper function to handle the minimizing player
        function MAX-VALUE(state, α, β) returns a utility value
        if TERMINAL-TEST(state) then return UTILITY(state)
        v ← −∞
        for each a in ACTIONS(state) do
            v ← MAX(v, MIN-VALUE(RESULT(s,a), α, β))
            if v ≥ β then return v
            α ← MAX(α, v)
        return v
        '''
        minimax_call_depth = minimax_call_depth + 1
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        if (board.is_winner(self) or board.is_loser(self) or (minimax_call_depth == depth) or (len(board.get_legal_moves()) == 0)):
            return self.score(board, self)

        value_return = float('-inf')
        for move in board.get_legal_moves():
            value_return = max(value_return, self.min_value_AB(board.forecast_move(move), alpha, beta, minimax_call_depth, depth))
            if (value_return >= beta):
                return value_return
            alpha = max(alpha, value_return)
        return value_return

    def min_value_AB(self, board, alpha, beta, minimax_call_depth, depth):
        '''
        Alphabeta Helper function to handle the minimizing player
        function MIN-VALUE(state, α, β) returns a utility value
        if TERMINAL-TEST(state) then return UTILITY(state)
        v ← +∞
        for each a in ACTIONS(state) do
            v ← MIN(v, MAX-VALUE(RESULT(s,a) , α, β))
            if v ≤ α then return v
            β ← MIN(β, v)
        return v
        '''
        minimax_call_depth = minimax_call_depth + 1
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()
        if (board.is_winner(self) or board.is_loser(self) or (minimax_call_depth == depth) or (len(board.get_legal_moves()) == 0)):
            return self.score(board, self)

        value_return = float('inf')
        for move in board.get_legal_moves():
            value_return = min(value_return, self.max_value_AB(board.forecast_move(move), alpha, beta, minimax_call_depth, depth))
            if (value_return <= alpha):
                return value_return
            beta = min(beta, value_return)
        return value_return


    def alphabeta(self, game, depth, alpha=float("-inf"), beta=float("inf"), maximizing_player=True):

        '''
        function ALPHA-BETA-SEARCH(state) returns an action
            v ← MAX-VALUE(state,−∞,+∞)
            return the action in ACTIONS(state) with value v

        '''

        """Implement minimax search with alpha-beta pruning as described in the
        lectures.

        Parameters
        ----------
        game : isolation.Board
            An instance of the Isolation game `Board` class representing the
            current game state

        depth : int
            Depth is an integer representing the maximum number of plies to
            search in the game tree before aborting

        alpha : float
            Alpha limits the lower bound of search on minimizing layers

        beta : float
            Beta limits the upper bound of search on maximizing layers

        maximizing_player : bool
            Flag indicating whether the current search depth corresponds to a
            maximizing layer (True) or a minimizing layer (False)

        Returns
        -------
        float
            The score for the current search branch

        tuple(int, int)
            The best move for the current branch; (-1, -1) for no legal moves

        Notes
        -----
            (1) You MUST use the `self.score()` method for board evaluation
                to pass the project unit tests; you cannot call any other
                evaluation function directly.
        """
        minimax_call_depth = 0
        score_and_move_tuple = []
        value_return = float('-inf')
        if self.time_left() < self.TIMER_THRESHOLD:
            raise Timeout()

        if(len(game.get_legal_moves()) == 0):
            return(self.score(game, self), (-1, -1))

        for move in game.get_legal_moves():
            next_state = game.forecast_move(move)
            value_return = self.min_value_AB(next_state, alpha, beta, minimax_call_depth, depth)
            score_and_move_tuple.append((value_return, move))
            alpha = max(alpha, value_return)

        return max(score_and_move_tuple, key = lambda t: t[0])


