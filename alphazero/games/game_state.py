from typing import Generic, List, Optional, TypeVar

from .move import Move
from .player import Player

M = TypeVar('M', bound=Move)
P = TypeVar('P', bound=Player)


class GameState(Generic[M, P]):
    def next(self, move: Move) -> 'GameState':
        raise NotImplementedError

    @property
    def player(self) -> P:
        raise NotImplementedError

    def get_legal_moves(self) -> List[M]:
        raise NotImplementedError

    def winner(self) -> Optional[P]:
        raise NotImplementedError

    def is_terminal(self) -> bool:
        raise NotImplementedError

    def is_win(self) -> bool:
        raise NotImplementedError

    def is_lose(self) -> bool:
        raise NotImplementedError

    def is_tie(self) -> bool:
        raise NotImplementedError

    def reverse_player(self) -> 'GameState':
        raise NotImplementedError
