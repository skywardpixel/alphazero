import enum

from alphazero.games.player import Player


class TicTacToePlayer(Player, enum.Enum):
    X = 0
    O = 1

    @property
    def opponent(self):
        return (TicTacToePlayer.X
                if self == TicTacToePlayer.O
                else TicTacToePlayer.O)

    def __str__(self):
        return self.name
