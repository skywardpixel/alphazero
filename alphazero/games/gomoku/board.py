import copy
from typing import Dict, Optional, List, Sequence

from alphazero.games.board import Board
from .exception import IllegalGomokuMoveException
from .move import GomokuMove
from .player import GomokuPlayer
from .zobrist_hash import EMPTY_BOARD, HASH_CODE


class GomokuBoard(Board):
    def __init__(self, size: int = 15, n: int = 5) -> None:
        self._size = size
        self.n = n
        self.grid: Dict[GomokuMove, GomokuPlayer] = dict()
        self._hash = EMPTY_BOARD

    @property
    def size(self):
        return self._size

    def apply_move(self, player: GomokuPlayer, move: GomokuMove) -> None:
        if not self.is_legal_move(move):
            raise IllegalGomokuMoveException
        self.grid[move] = player
        self._hash ^= HASH_CODE[(move, player)]

    def get(self, r: int, c: int) -> Optional[GomokuPlayer]:
        return self.grid.get(GomokuMove(r, c))

    def get_legal_moves(self, within: Optional[int] = None) -> List[GomokuMove]:
        if within is not None:
            moves = set()
            for p in self.grid:
                for r in range(p.x - within, p.x + within + 1):
                    for c in range(p.y - within, p.y + within + 1):
                        move = GomokuMove(r, c)
                        if self.is_legal_move(move):
                            moves.add(move)
            return list(moves)
        else:
            return [GomokuMove(r, c)
                    for r in range(self.size) for c in range(self.size)
                    if self.get(r, c) is None]

    def is_legal_move(self, move: GomokuMove) -> bool:
        return 0 <= move.x < self.size \
               and 0 <= move.y < self.size \
               and self.grid.get(move) is None

    def has_won(self, player: GomokuPlayer) -> bool:
        return self._has_win_in_a_row(player) \
               or self._has_win_in_a_column(player) \
               or self._has_win_in_a_diagonal(player)

    def _has_win_in_a_row(self, player: GomokuPlayer):
        for r in range(self.size):
            row = [self.get(r, c) for c in range(self.size)]
            if self._has_consecutive_n(row, player):
                return True
        return False

    def _has_win_in_a_column(self, player: GomokuPlayer):
        for c in range(self.size):
            column = [self.get(r, c) for r in range(self.size)]
            if self._has_consecutive_n(column, player):
                return True
        return False

    def _has_win_in_a_diagonal(self, player: GomokuPlayer):
        # major direction: top left -> bottom right
        for i in range(2 * self.size - 1):
            # i = 0        -> point in bottom left
            # i = size-1   -> major diagonal
            # i = 2*size-1 -> point in top right
            if i < self.size:
                # below major diagonal, r - c = size - 1 - i
                # r_first = size - 1 - i, r_last = size - 1
                diagonal = [self.get(r, r - i)
                            for r in range(self.size - 1 - i, self.size)]
            else:
                # above major diagonal, c - r = i - size + 1
                # r_first = 0, r_last = 2 * size - i - 2
                diagonal = [self.get(r, r + i - self.size + 1)
                            for r in range(0, 2 * self.size - i - 1)]

            if self._has_consecutive_n(diagonal, player):
                return True

        # minor direction: top right -> bottom left
        for i in range(2 * self.size - 1):
            # i = 0        -> point in top left
            # i = size-1   -> minor diagonal
            # i = 2size-1 -> point in bottom right
            if i < self.size:
                # above minor diagonal, r + c = i
                # r_first = 0, r_last = i
                diagonal = [self.get(r, i - r)
                            for r in range(0, i + 1)]
            else:
                # below minor diagonal, r + c = i
                # r_first = i - size + 1, r_last = size - 1
                diagonal = [self.get(r, i - r)
                            for r in range(i - self.size + 1, self.size)]
            if self._has_consecutive_n(diagonal, player):
                return True

        return False

    def _has_consecutive_n(self, sequence: Sequence[GomokuPlayer], player: GomokuPlayer):
        if len(sequence) < self.n:
            return False
        for left in range(len(sequence) - self.n + 1):
            window = sequence[left:left + self.n]
            if all(x == player for x in window):
                return True
        return False

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

    def copy(self) -> 'GomokuBoard':
        return copy.deepcopy(self)

    def zobrist_hash(self) -> int:
        return self._hash
