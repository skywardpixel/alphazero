from dataclasses import dataclass

from alphazero.games.move import Move


@dataclass(eq=True, frozen=True)
class GomokuMove(Move):
    x: int
    y: int

    def __str__(self):
        return f'({self.x},{self.y})'
