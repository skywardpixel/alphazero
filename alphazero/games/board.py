from typing import Optional

from .player import Player


class Board:
    @property
    def size(self) -> int:
        raise NotImplementedError

    def get(self, r: int, c: int) -> Optional[Player]:
        raise NotImplementedError

    def zobrist_hash(self) -> int:
        raise NotImplementedError
