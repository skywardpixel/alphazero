import copy
import enum
from dataclasses import dataclass
from typing import Dict, Optional, List, AbstractSet

from alphazero.games.base import IllegalMoveException, Player, GameState, Move, Game
from alphazero.games.go.zobrist_hashing import EMPTY_BOARD, HASH_CODE


class GoPlayer(Player, enum.Enum):
    BLACK = 0
    WHITE = 1

    @property
    def opponent(self):
        return (GoPlayer.BLACK
                if self == GoPlayer.WHITE
                else GoPlayer.WHITE)

    def __str__(self):
        return '●' if self == GoPlayer.BLACK else '○'


@dataclass(eq=True, frozen=True)
class GoPoint:
    x: int
    y: int

    def neighbors(self):
        return [
            GoPoint(self.x - 1, self.y),
            GoPoint(self.x + 1, self.y),
            GoPoint(self.x, self.y - 1),
            GoPoint(self.x, self.y + 1),
        ]

    def __str__(self):
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
    def is_play(self) -> bool:
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

    @classmethod
    def play(cls, x, y) -> 'GoMove':
        return GoMove(GoPoint(x, y))

    @classmethod
    def pass_turn(cls) -> 'GoMove':
        return GoMove(is_pass=True)

    @classmethod
    def resign(cls) -> 'GoMove':
        return GoMove(is_resign=True)


class GoString:
    def __init__(self,
                 player: GoPlayer,
                 stones: AbstractSet[GoPoint],
                 liberties: AbstractSet[GoPoint]):
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
        self._grid: Dict[GoPoint, GoString] = dict()
        self._hash = EMPTY_BOARD

    def place_stone(self, player: GoPlayer, point: GoPoint) -> None:
        if not self.is_legal_point(point):
            raise IllegalGoMoveException
        adjacent_friendly = set()
        adjacent_enemy = set()
        liberties = set()
        for neighbor in point.neighbors():
            if self.is_within_bounds(neighbor):
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    liberties.add(neighbor)
                elif neighbor_string.player == player:
                    adjacent_friendly.add(neighbor_string)
                else:
                    adjacent_enemy.add(neighbor_string)
        new_string = GoString(player, {point}, liberties)

        for friendly in adjacent_friendly:
            new_string = new_string.merge(friendly)
        for point in new_string:
            self._grid[point] = new_string

        # apply hash code
        self._hash ^= HASH_CODE[point, player]

        for enemy in adjacent_enemy:
            new_enemy = enemy.remove_liberty(point)
            if new_enemy.liberties:
                self._replace_string(new_enemy)
            else:
                self._remove_string(enemy)

    def _replace_string(self, new_string: GoString) -> None:
        for point in new_string.stones:
            self._grid[point] = new_string

    def _remove_string(self, string: GoString) -> None:
        for point in string.stones:
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    self._replace_string(neighbor_string.add_liberty(point))
            self._grid[point] = None

            self._hash ^= HASH_CODE[point, string.player]  # <3>

    def get_string(self, r: int, c: int) -> Optional[GoString]:
        return self._grid.get(GoPoint(r, c))

    def get(self, r: int, c: int) -> Optional[GoPlayer]:
        string = self.get_string(r, c)
        return string.player if string is not None else None

    def get_legal_points(self) -> List[GoPoint]:
        return [GoPoint(r, c)
                for r in range(self.size) for c in range(self.size)
                if self.get(r, c) is None]

    def is_legal_point(self, point: GoPoint) -> bool:
        return self.is_within_bounds(point) and self._grid.get(point) is None

    def is_within_bounds(self, point: GoPoint):
        return 0 <= point.x < self.size and 0 <= point.y < self.size

    def has_won(self, player: GoPlayer) -> bool:
        raise NotImplementedError

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
            next_board.place_stone(self.player, move.point)
            return GoGameState(next_board, self.player.opponent)

    def get_legal_moves(self) -> List[GoMove]:
        return [GoMove(p) for p in self.board.get_legal_points()] \
               + [GoMove.pass_turn(), GoMove.resign()]

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
