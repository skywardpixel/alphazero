import enum

from alphazero.games.player import Player


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
