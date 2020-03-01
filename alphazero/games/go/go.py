import copy
import enum
from dataclasses import dataclass
from typing import Dict, Collection, Optional, List

from alphazero.games.base import IllegalMoveException, Player, GameState, Move, Game


class GoPlayer(Player, enum.Enum):
    BLACK = 0
    WHITE = 1

    @property
    def opponent(self):
        return (GoPlayer.BLACK
                if self == GoPlayer.WHITE
                else GoPlayer.WHITE)

    def __repr__(self):
        return '●' if self == GoPlayer.BLACK else '○'


@dataclass(eq=True, frozen=True)
class GoPoint:
    x: int
    y: int

    def __repr__(self):
        return f"({self.x},{self.y})"


class GoMove(Move):
    def __init__(self,
                 point: Optional[GoPoint] = None,
                 is_pass: bool = False,
                 is_resign: bool = True):
        if point is not None and (is_pass or is_resign):
            raise Exception
        elif is_pass and is_resign:
            raise Exception
        self.point = point
        self.is_pass = is_pass
        self.is_resign = is_resign

    @property
    def is_set_stone(self) -> bool:
        return self.point is not None

    @property
    def x(self) -> int:
        if self.point is None:
            raise Exception('Move is resign or pass')
        return self.point.x

    @property
    def y(self) -> int:
        if self.point is None:
            raise Exception('Move is resign or pass')
        return self.point.y


class GoString:
    def __init__(self,
                 player: GoPlayer,
                 stones: Collection[GoPlayer],
                 liberties: Collection[GoMove]):
        self.player = player
        self.stones = frozenset(stones)
        self.liberties = frozenset(liberties)

    def add_liberty(self, point: GoPoint):
        return GoString(self.player, self.stones, self.liberties | {point})

    def remove_liberty(self, point: GoPoint):
        return GoString(self.player, self.stones, self.liberties - {point})

    def merge(self, other: 'GoString'):
        if self.player != other.player:
            raise Exception('GoStrings of different players can\'t be merged')
        combined_stones = self.stones | other.stones
        return GoString(self.player,
                        combined_stones,
                        (self.liberties | other.liberties) - combined_stones)

    def num_liberties(self):
        return len(self.liberties)


class GoBoard:
    def __init__(self, size: int = 15) -> None:
        self.size = size
        self.grid: Dict[GoPoint, GoPlayer] = dict()

    def apply_move(self, player: GoPlayer, move: GoMove) -> None:
        if not self.is_legal_move(move):
            raise IllegalGoMoveException
        self.grid[move] = player

    def get(self, r: int, c: int) -> Optional[GoPlayer]:
        return self.grid.get(GoPoint(r, c))

    def get_legal_points(self) -> List[GoPoint]:
        return [GoPoint(r, c)
                for r in range(self.size) for c in range(self.size)
                if self.get(r, c) is None]

    def is_legal_move(self, move: GoMove) -> bool:
        if not move.is_set_stone:
            return True
        return 0 <= move.x < self.size \
               and 0 <= move.y < self.size \
               and self.grid.get(move) is None

    def has_won(self, player: GoPlayer) -> bool:
        raise NotImplementedError

    def full(self) -> bool:
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

    def copy(self) -> 'GoBoard':
        return copy.deepcopy(self)


class GoGameState(GameState):
    def __init__(self,
                 board: GoBoard,
                 player: GoPlayer,
                 resigned: Optional[GoPlayer] = None) -> None:
        super().__init__()
        self.board = board
        self.player = player
        self.resigned = resigned

    @classmethod
    def get_initial_state(cls, size: int = 9):
        return GoGameState(GoBoard(size), GoPlayer.BLACK)

    @property
    def current_player(self) -> Player:
        return self.player

    def next(self, move: GoMove) -> 'GoGameState':
        if self.is_terminal():
            raise IllegalGoMoveException("Game has ended.")
        next_board = self.board.copy()
        if move.is_resign:
            return GoGameState(next_board, self.player.opponent, resigned=self.player)
        elif move.is_pass:
            return GoGameState(next_board, self.player.opponent)
        else:
            next_board.apply_move(self.player, move)
            return GoGameState(next_board, self.player.opponent)

    def get_legal_moves(self) -> List[GoMove]:
        return [GoMove(p) for p in self.board.get_legal_points()] \
               + [GoMove(is_pass=True), GoMove(is_pass=True)]

    def winner(self) -> Optional[GoPlayer]:
        if self.resigned is not None:
            return self.resigned.opponent
        if self.board.has_won(GoPlayer.BLACK):
            return GoPlayer.BLACK
        elif self.board.has_won(GoPlayer.WHITE):
            return GoPlayer.WHITE
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
        return GoGameState(self.board.copy(), self.player.opponent)


class GoGame(Game):
    def __init__(self, size: int = 9):
        super().__init__()
        self.state = GoGameState.get_initial_state(size)

    def play(self, move: GoMove):
        self.state = self.state.next(move)

    def show_board(self) -> None:
        print(self.state.board)

    @property
    def is_over(self) -> bool:
        return self.state.is_terminal()

    @property
    def current_player(self) -> GoPlayer:
        return self.state.player

    @property
    def winner(self) -> GoPlayer:
        return self.state.winner()


class IllegalGoMoveException(IllegalMoveException):
    pass
