from typing import List, Optional

from alphazero.games import GameState, Player
from .board import GomokuBoard
from .exception import IllegalGomokuMoveException
from .move import GomokuMove
from .player import GomokuPlayer


class GomokuGameState(GameState[GomokuMove, GomokuPlayer, GomokuBoard]):

    def __init__(self, board: GomokuBoard, player: GomokuPlayer) -> None:
        super().__init__()
        self._board = board
        self._player = player

    @property
    def player(self) -> GomokuPlayer:
        return self._player

    @property
    def board(self) -> GomokuBoard:
        return self._board

    @classmethod
    def get_initial_state(cls, size: int = 15, n: int = 5):
        return GomokuGameState(GomokuBoard(size, n), GomokuPlayer.BLACK)

    @property
    def current_player(self) -> Player:
        return self.player

    def next(self, move: GomokuMove) -> 'GomokuGameState':
        if self.is_terminal():
            raise IllegalGomokuMoveException("Game has ended.")
        next_board = self.board.copy()
        next_board.apply_move(self.player, move)
        return GomokuGameState(next_board, self.player.opponent)

    def get_legal_moves(self) -> List[GomokuMove]:
        return self.board.get_legal_moves(within=2)

    def is_legal_move(self, move: GomokuMove) -> bool:
        return self.board.is_legal_move(move)

    def winner(self) -> Optional[GomokuPlayer]:
        if self.board.has_won(GomokuPlayer.BLACK):
            return GomokuPlayer.BLACK
        elif self.board.has_won(GomokuPlayer.WHITE):
            return GomokuPlayer.WHITE
        else:
            return None

    def is_terminal(self) -> bool:
        return self.is_win() or self.is_lose() or self.is_tie()

    def is_win(self) -> bool:
        return self.winner() == self.player

    def is_lose(self) -> bool:
        return self.winner() == self.player.opponent

    def is_tie(self) -> bool:
        return self.winner() is None and self.board.full()

    def canonical(self) -> 'GomokuGameState':
        if self.player == GomokuPlayer.BLACK:
            return self
        rev_board = self.board.copy()
        for point, player in self.board.grid.items():
            rev_board.grid[point] = player.opponent
        return GomokuGameState(rev_board, self.player.opponent)
