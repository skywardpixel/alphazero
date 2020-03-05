from typing import Generic, List, Optional, TypeVar, Tuple

from .move import Move
from .player import Player

M = TypeVar('M', bound=Move)
P = TypeVar('P', bound=Player)


class GameState(Generic[M, P]):
    canonical_player: Player

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

    def canonical(self) -> 'GameState':
        raise NotImplementedError

    def compact(self) -> Tuple[P, int]:
        raise NotImplementedError

    def board_zobrist_hash(self) -> int:
        raise NotImplementedError
