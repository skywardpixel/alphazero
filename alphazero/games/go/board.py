import copy
from typing import Dict, Optional, List

from .exception import IllegalGoMoveException
from .player import GoPlayer
from .point import GoPoint
from .string import GoString
from .zobrist_hashing import EMPTY_BOARD, HASH_CODE


class GoBoard:
    def __init__(self, size: int = 15) -> None:
        self.size = size
        self._grid: Dict[GoPoint, GoString] = dict()
        self._hash = EMPTY_BOARD

    def place_stone(self, player: GoPlayer, point: GoPoint) -> None:
        if not self.is_empty_point(point):
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
        for p in new_string.stones:
            self._grid[p] = new_string

        # apply hash code
        self._hash ^= HASH_CODE[point, player]

        for enemy in adjacent_enemy:
            new_enemy = enemy.remove_liberty(point)
            if new_enemy.liberties:
                self._update_string(new_enemy)
            else:
                self._remove_string(enemy)

    def _update_string(self, new_string: GoString) -> None:
        for point in new_string.stones:
            self._grid[point] = new_string

    def _remove_string(self, string: GoString) -> None:
        for point in string.stones:
            for neighbor in point.neighbors():
                neighbor_string = self._grid.get(neighbor)
                if neighbor_string is None:
                    continue
                if neighbor_string is not string:
                    self._update_string(neighbor_string.add_liberty(point))
            self._grid[point] = None

            self._hash ^= HASH_CODE[point, string.player]  # <3>

    def get_string(self, r: int, c: int) -> Optional[GoString]:
        return self._grid.get(GoPoint(r, c))

    def get(self, r: int, c: int) -> Optional[GoPlayer]:
        string = self.get_string(r, c)
        return string.player if string is not None else None

    def get_empty_points(self) -> List[GoPoint]:
        return [GoPoint(r, c)
                for r in range(self.size) for c in range(self.size)
                if self.get(r, c) is None]

    def is_empty_point(self, point: GoPoint) -> bool:
        return self.is_within_bounds(point) and self._grid.get(point) is None

    def is_move_self_capture(self, player: GoPlayer, point: GoPoint):
        next_board = self.copy()
        next_board.place_stone(player, point)
        new_string = next_board.get_string(point.x, point.y)
        return new_string.num_liberties == 0

    def is_within_bounds(self, point: GoPoint):
        return 0 <= point.x < self.size and 0 <= point.y < self.size

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

    def zobrist_hash(self) -> int:
        return self._hash
