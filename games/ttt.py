import enum
from collections import namedtuple
from typing import List, Dict, Optional, Union

from games.base import *


class TicTacToePlayer(Player, enum.Enum):
    X = 0
    O = 1

    @property
    def opponent(self):
        return (TicTacToePlayer.X
                if self == TicTacToePlayer.O
                else TicTacToePlayer.O)


class TicTacToeMove(Move, namedtuple('TicTacToeMove', ['x', 'y'])):
    def __repr__(self):
        return f"({self.x},{self.y})"


class TicTacToeBoard:
    def __init__(self, size: int) -> None:
        self.size = size
        self.board: Dict[TicTacToeMove, TicTacToePlayer] = dict()

    def play(self, player: TicTacToePlayer, point: TicTacToeMove) -> None:
        if not self.is_legal_move(point):
            raise IllegalTicTacToeMoveException
        self.board[point] = player

    def get(self, r: int, c: int) -> Optional[TicTacToePlayer]:
        return self.board.get((r, c))

    def get_legal_moves(self) -> List[TicTacToeMove]:
        return [TicTacToeMove(r, c)
                for r in range(self.size) for c in range(self.size)
                if self.get(r, c) is None]

    def is_legal_move(self, move: TicTacToeMove) -> bool:
        return 0 <= move.x < self.size \
               and 0 <= move.y < self.size \
               and self.board.get(move) is None

    def _has_full_row(self, player: TicTacToePlayer) -> bool:
        return any(all(self.get(r, c) == player for c in range(self.size))
                   for r in range(self.size))

    def _has_full_column(self, player: TicTacToePlayer) -> bool:
        return any(all(self.get(r, c) == player for r in range(self.size))
                   for c in range(self.size))

    def _has_full_diagonal(self, player: TicTacToePlayer) -> bool:
        major = all(self.get(i, i) == player
                    for i in range(self.size))
        minor = all(self.get(i, self.size - 1 - i) == player
                    for i in range(self.size))
        return major or minor

    def has_won(self, player: TicTacToePlayer) -> bool:
        return self._has_full_row(player) \
               or self._has_full_column(player) \
               or self._has_full_diagonal(player)

    def is_full(self) -> bool:
        return all(self.get(r, c) is not None
                   for r in range(self.size)
                   for c in range(self.size))

    def show(self) -> None:
        for r in range(self.size):
            print([self._show_point(r, c) for c in range(self.size)])

    def _show_point(self, r: int, c: int) -> str:
        player = self.get(r, c)
        return player.name if player is not None else '-'


class TicTacToeGameState(GameState):
    def __init__(self, next_player: TicTacToePlayer) -> None:
        self.board = TicTacToeBoard(3)
        self.next_player = next_player

    def play(self, move: TicTacToeMove) -> None:
        if self.is_terminal():
            raise IllegalTicTacToeMoveException("Game has ended.")
        self.board.play(self.next_player, move)
        self.next_player = self.next_player.opponent

    def get_legal_moves(self) -> List[TicTacToeMove]:
        return self.board.get_legal_moves()

    def is_terminal(self) -> bool:
        return self.winner() is not None or self.board.is_full()

    def winner(self) -> Optional[TicTacToePlayer]:
        if self.board.has_won(TicTacToePlayer.X):
            return TicTacToePlayer.X
        elif self.board.has_won(TicTacToePlayer.O):
            return TicTacToePlayer.O
        else:
            return None


class TicTacToeGame(Game):
    def __init__(self):
        super().__init__()
        self.state = TicTacToeGameState(TicTacToePlayer.X)

    def play(self, move: TicTacToeMove):
        self.state.play(move)

    @property
    def is_terminal(self) -> bool:
        return self.state.is_terminal()

    def show_board(self) -> None:
        self.state.board.show()

    @property
    def next_player(self) -> TicTacToePlayer:
        return self.state.next_player

    @property
    def winner(self) -> TicTacToePlayer:
        return self.state.winner()


class IllegalTicTacToeMoveException(IllegalMoveException):
    pass
