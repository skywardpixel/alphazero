import copy
import enum
from dataclasses import dataclass
from typing import Dict

from games.base import *


class TicTacToePlayer(Player, enum.Enum):
    X = 0
    O = 1

    @property
    def opponent(self):
        return (TicTacToePlayer.X
                if self == TicTacToePlayer.O
                else TicTacToePlayer.O)

    def __repr__(self):
        return self.name


@dataclass(eq=True, frozen=True)
class TicTacToeMove(Move):
    x: int
    y: int

    def __repr__(self):
        return f"({self.x},{self.y})"


class TicTacToeBoard:
    def __init__(self, size: int) -> None:
        self.size = size
        self.grid: Dict[TicTacToeMove, TicTacToePlayer] = dict()

    def play(self, player: TicTacToePlayer, move: TicTacToeMove) -> None:
        if not self.is_legal_move(move):
            raise IllegalTicTacToeMoveException
        self.grid[move] = player

    def get(self, r: int, c: int) -> Optional[TicTacToePlayer]:
        return self.grid.get(TicTacToeMove(r, c))

    def get_legal_moves(self) -> List[TicTacToeMove]:
        return [TicTacToeMove(r, c)
                for r in range(self.size) for c in range(self.size)
                if self.get(r, c) is None]

    def is_legal_move(self, move: TicTacToeMove) -> bool:
        return 0 <= move.x < self.size \
               and 0 <= move.y < self.size \
               and self.grid.get(move) is None

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
        # return self._has_full_diagonal(player)

    def is_full(self) -> bool:
        return all(self.get(r, c) is not None
                   for r in range(self.size)
                   for c in range(self.size))

    def __repr__(self) -> str:
        result = ''
        for r in range(self.size):
            result += ' '.join([self._show_point(r, c) for c in range(self.size)])
            result += '\n'
        return result

    def _show_point(self, r: int, c: int) -> str:
        player = self.get(r, c)
        return repr(player) if player is not None else '-'

    def copy(self) -> 'TicTacToeBoard':
        return copy.deepcopy(self)


class TicTacToeGameState(GameState):
    def __init__(self, board: TicTacToeBoard, player: TicTacToePlayer) -> None:
        super().__init__()
        self.board = board
        self.player = player

    @classmethod
    def get_initial_state(self):
        return TicTacToeGameState(TicTacToeBoard(3), TicTacToePlayer.X)

    @property
    def current_player(self) -> Player:
        return self.player

    def next(self, move: TicTacToeMove) -> 'TicTacToeGameState':
        if self.is_terminal():
            raise IllegalTicTacToeMoveException("Game has ended.")
        next_board = self.board.copy()
        next_board.play(self.player, move)
        return TicTacToeGameState(next_board, self.player.opponent)

    def get_legal_moves(self) -> List[TicTacToeMove]:
        return self.board.get_legal_moves()

    def winner(self) -> Optional[TicTacToePlayer]:
        if self.board.has_won(TicTacToePlayer.X):
            return TicTacToePlayer.X
        elif self.board.has_won(TicTacToePlayer.O):
            return TicTacToePlayer.O
        else:
            return None

    def is_terminal(self) -> bool:
        return self.is_win() or self.is_lose() or self.is_tie()

    def is_win(self) -> bool:
        return self.winner() == self.player

    def is_lose(self) -> bool:
        return self.winner() == self.player.opponent

    def is_tie(self) -> bool:
        return self.winner() is None and self.board.is_full()

    def reverse_player(self) -> 'GameState':
        # reversed_board = self.board.copy()
        # for p in self.board.grid:
        #     reversed_board.grid[p] = self.board.grid[p].opponent
        return TicTacToeGameState(self.board.copy(), self.player.opponent)


class TicTacToeGame(Game):
    def __init__(self):
        super().__init__()
        self.state = TicTacToeGameState.get_initial_state()

    def play(self, move: TicTacToeMove):
        self.state = self.state.next(move)

    def show_board(self) -> None:
        print(self.state.board)

    @property
    def is_over(self) -> bool:
        return self.state.is_terminal()

    @property
    def current_player(self) -> TicTacToePlayer:
        return self.state.player

    @property
    def winner(self) -> TicTacToePlayer:
        return self.state.winner()


class IllegalTicTacToeMoveException(IllegalMoveException):
    pass
