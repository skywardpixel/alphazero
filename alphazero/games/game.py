from typing import TypeVar, Generic

from .game_state import GameState
from .move import Move
from .player import Player

S = TypeVar('S', bound=GameState)
M = TypeVar('M', bound=Move)
P = TypeVar('P', bound=Player)


class Game(Generic[S, M, P]):

    def play(self, move: M) -> None:
        raise NotImplementedError

    @property
    def state(self) -> S:
        raise NotImplementedError

    @property
    def is_over(self) -> bool:
        raise NotImplementedError

    def show_board(self) -> None:
        raise NotImplementedError

    @property
    def current_player(self) -> P:
        raise NotImplementedError

    @property
    def winner(self) -> P:
        raise NotImplementedError

    @property
    def action_space_size(self) -> int:
        raise NotImplementedError

    def reset(self) -> None:
        raise NotImplementedError

    def move_to_index(self, move: M) -> int:
        """
        Return the index of the move.
        `0` is reserved for pass, `size` is reserved for resign.
        :param move: move to return index of
        :return: the index of the move
        """
        raise NotImplementedError

    def index_to_move(self, index: int) -> M:
        """
        Return the move corresponding to the index.
        Inverse of move_to_index.
        :param index: the index of the move
        :return: the move corresponding to the index
        """
        raise NotImplementedError
