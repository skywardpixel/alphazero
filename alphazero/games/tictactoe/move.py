from dataclasses import dataclass

from alphazero.games.base import Move


@dataclass(eq=True, frozen=True)
class TicTacToeMove(Move):
    x: int
    y: int

    def __str__(self):
        return f"({self.x},{self.y})"
