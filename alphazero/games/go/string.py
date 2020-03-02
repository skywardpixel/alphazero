from typing import AbstractSet

from .point import GoPoint
from .player import GoPlayer


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

    @property
    def num_liberties(self):
        return len(self.liberties)
