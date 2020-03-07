from dataclasses import dataclass

from alphazero.games.move import Move


@dataclass(eq=True, frozen=True)
class TicTacToeMove(Move):
    x: int
    y: int

    def __str__(self):
        return f"({self.x},{self.y})"

    @classmethod
    def from_string(cls, string: str) -> 'TicTacToeMove':
        string = string.lower().strip()
        x, y = string.split()
        return TicTacToeMove(int(x), int(y))
