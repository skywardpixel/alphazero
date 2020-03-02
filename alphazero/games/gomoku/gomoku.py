import copy
import enum
from dataclasses import dataclass
from typing import Dict, Sequence, Optional, List

from alphazero.games.base import Move, Player, GameState, Game, IllegalMoveException


class GomokuPlayer(Player, enum.Enum):
    BLACK = 0
    WHITE = 1

    @property
    def opponent(self):
        return (GomokuPlayer.BLACK
                if self == GomokuPlayer.WHITE
                else GomokuPlayer.WHITE)

    def __str__(self):
        return '●' if self == GomokuPlayer.BLACK else '○'


@dataclass(eq=True, frozen=True)
class GomokuMove(Move):
    x: int
    y: int

    def __str__(self):
        return f'({self.x},{self.y})'


class GomokuBoard:
    def __init__(self, size: int = 15, n: int = 5) -> None:
        self.size = size
        self.n = n
        self._grid: Dict[GomokuMove, GomokuPlayer] = dict()

    def apply_move(self, player: GomokuPlayer, move: GomokuMove) -> None:
        if not self.is_legal_move(move):
            raise IllegalGomokuMoveException
        self._grid[move] = player

    def get(self, r: int, c: int) -> Optional[GomokuPlayer]:
        return self._grid.get(GomokuMove(r, c))

    def get_legal_moves(self, within: Optional[int] = None) -> List[GomokuMove]:
        if within is not None:
            moves = set()
            for p in self._grid:
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
               and self._grid.get(move) is None

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


class GomokuGameState(GameState):
    def __init__(self, board: GomokuBoard, player: GomokuPlayer) -> None:
        super().__init__()
        self.board = board
        self.player = player

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

    def reverse_player(self) -> 'GameState':
        # reversed_board = self.board.copy()
        # for p in self.board.grid:
        #     reversed_board.grid[p] = self.board.grid[p].opponent
        return GomokuGameState(self.board.copy(), self.player.opponent)


class GomokuGame(Game):
    def __init__(self, size: int = 15, n: int = 5):
        super().__init__()
        self.state = GomokuGameState.get_initial_state(size, n)

    def play(self, move: GomokuMove):
        self.state = self.state.next(move)

    def show_board(self) -> None:
        print(self.state.board)

    @property
    def is_over(self) -> bool:
        return self.state.is_terminal()

    @property
    def current_player(self) -> GomokuPlayer:
        return self.state.player

    @property
    def winner(self) -> GomokuPlayer:
        return self.state.winner()


class IllegalGomokuMoveException(IllegalMoveException):
    pass
