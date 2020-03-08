from typing import Generic, List, Optional, TypeVar

from .board import Board
from .move import Move
from .player import Player

M = TypeVar('M', bound=Move)
P = TypeVar('P', bound=Player)
B = TypeVar('B', bound=Board)


class GameState(Generic[M, P, B]):
    def next(self, move: Move) -> 'GameState':
        raise NotImplementedError

    @property
    def player(self) -> P:
        raise NotImplementedError

    @property
    def board(self) -> B:
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

    def canonical(self) -> 'GameState':
        raise NotImplementedError
