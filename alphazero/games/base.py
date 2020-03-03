from typing import List, Optional


class Player:
    @property
    def opponent(self):
        raise NotImplementedError


class Move:
    pass


class Game:
    def play(self, move: Move):
        raise NotImplementedError

    @property
    def is_over(self) -> bool:
        raise NotImplementedError

    def show_board(self) -> None:
        raise NotImplementedError

    @property
    def current_player(self) -> Player:
        raise NotImplementedError

    @property
    def winner(self) -> Player:
        raise NotImplementedError

    @property
    def action_space_size(self) -> int:
        raise NotImplementedError

    def move_to_index(self, move: Move) -> int:
        """
        Return the index of the move.
        `0` is reserved for pass, `size` is reserved for resign.
        :param move: move to return index of
        :return: the index of the move
        """
        raise NotImplementedError

    def index_to_move(self, index: int) -> Move:
        """
        Return the move corresponding to the index.
        Inverse of move_to_index.
        :param index: the index of the move
        :return: the move corresponding to the index
        """
        raise NotImplementedError


class GameState:
    def next(self, move: Move) -> 'GameState':
        raise NotImplementedError

    @property
    def current_player(self) -> Player:
        raise NotImplementedError

    def get_legal_moves(self) -> List[Move]:
        raise NotImplementedError

    def winner(self) -> Optional[Player]:
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


class IllegalMoveException(Exception):
    pass
