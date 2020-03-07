from typing import Optional, List

from alphazero.games import GameState, Player
from .board import GoBoard
from .exception import IllegalGoMoveException
from .move import GoMove
from .player import GoPlayer
from .scoring import compute_game_result


class GoGameState(GameState[GoMove, GoPlayer, GoBoard]):
    canonical_player = GoPlayer.BLACK

    def __init__(self,
                 board: GoBoard,
                 player: GoPlayer,
                 previous_state: Optional['GoGameState'],
                 last_move: Optional[GoMove]) -> None:
        super().__init__()
        self._board = board
        self._player = player
        self.previous_state = previous_state
        if self.previous_state is None:
            self.previous_states = frozenset()
        else:
            self.previous_states = frozenset(
                previous_state.previous_states |
                {(previous_state.player, previous_state.board.zobrist_hash())}
            )
        self.last_move = last_move

    @property
    def player(self) -> GoPlayer:
        return self._player

    @property
    def board(self) -> GoBoard:
        return self._board

    @classmethod
    def get_initial_state(cls, size: int = 9) -> 'GoGameState':
        return cls(GoBoard(size), GoPlayer.BLACK, None, None)

    @property
    def current_player(self) -> Player:
        return self.player

    def next(self, move: GoMove) -> 'GoGameState':
        if self.is_terminal():
            raise IllegalGoMoveException("Game has ended.")
        next_board = self.board.copy()
        if move.is_play:
            next_board.place_stone(self.player, move.point)
        return GoGameState(next_board, self.player.opponent, self, move)

    def get_legal_moves(self) -> List[GoMove]:
        legal_play_moves = [GoMove(p)
                            for p in self.board.get_empty_points()
                            if self.is_legal_move(GoMove(p))]
        return legal_play_moves + [GoMove.resign(), GoMove.pass_turn()]

    def is_legal_move(self, move: GoMove):
        if self.is_terminal():
            return False
        if move.is_resign or move.is_pass:
            return True
        return self.board.get(move.x, move.y) is None \
               and not self.board.is_move_self_capture(self.player, move.point) \
               and not self.does_move_violate_ko(self.player, move)

    def does_move_violate_ko(self, player, move):
        if not move.is_play:
            return False
        next_board = self.board.copy()
        next_board.place_stone(player, move.point)
        next_state = (player.opponent, next_board.zobrist_hash())
        return next_state in self.previous_states

    def winner(self) -> Optional[GoPlayer]:
        if not self.is_terminal():
            return None
        if self.last_move.is_resign:
            return self.player
        game_result = compute_game_result(self)
        return game_result.winner

    def is_terminal(self) -> bool:
        if self.last_move is None:
            # initial state
            return False
        if self.last_move.is_resign:
            return True
        if self.last_move.is_pass:
            second_to_last_move = self.previous_state.last_move
            if second_to_last_move is not None and second_to_last_move.is_pass:
                return True
        return False

    def is_win(self) -> bool:
        return self.winner() == self.player

    def is_lose(self) -> bool:
        return self.winner() == self.player.opponent

    def is_tie(self) -> bool:
        return self.winner() is None and self.is_terminal()

    def canonical(self) -> 'GoGameState':
        """
        Regardless of the current player, returns the equivalent
        of the current state as if BLACK is playing.
        This is used for agents.
        :return:
        """
        if self.player == GoPlayer.BLACK:
            return self
        rev_board = self.board.copy()
        for string in set(rev_board.grid.values()):
            if string is not None:
                string.player = string.player.opponent
        return GoGameState(rev_board,
                           self.player.opponent,
                           self.previous_state,
                           self.last_move)
