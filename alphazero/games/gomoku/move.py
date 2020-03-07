from dataclasses import dataclass

from alphazero.games.move import Move


@dataclass(eq=True, frozen=True)
class GomokuMove(Move):
    x: int
    y: int

    def __str__(self):
        return f'({self.x},{self.y})'

    @classmethod
    def from_string(cls, string: str) -> 'GomokuMove':
        string = string.lower().strip()
        x, y = string.split()
        return GomokuMove(int(x), int(y))
