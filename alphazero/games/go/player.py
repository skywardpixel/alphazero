import enum

from alphazero.games.player import Player


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
