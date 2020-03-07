import copy
from typing import Dict, Optional, List

from alphazero.games.board import Board
from .exception import IllegalTicTacToeMoveException
from .move import TicTacToeMove
from .player import TicTacToePlayer
from .zobrist_hash import EMPTY_BOARD, HASH_CODE


class TicTacToeBoard(Board):
    def __init__(self, size: int = 3) -> None:
        self._size = size
        self.grid: Dict[TicTacToeMove, TicTacToePlayer] = dict()
        self._hash = EMPTY_BOARD

    @property
    def size(self):
        return self._size

    def apply_move(self, player: TicTacToePlayer, move: TicTacToeMove) -> None:
        if not self.is_legal_move(move):
            raise IllegalTicTacToeMoveException
        self.grid[move] = player
        self._hash ^= HASH_CODE[(move, player)]

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

    def full(self) -> bool:
        return all(self.get(r, c) is not None
                   for r in range(self.size)
                   for c in range(self.size))

    def __str__(self) -> str:
        result = ''
        for r in range(self.size):
            result += ' '.join([self._show_point(r, c) for c in range(self.size)])
            result += '\n'
        return result

    def _show_point(self, r: int, c: int) -> str:
        player = self.get(r, c)
        return str(player) if player is not None else '-'

    def copy(self) -> 'TicTacToeBoard':
        return copy.deepcopy(self)

    def zobrist_hash(self) -> int:
        return self._hash
