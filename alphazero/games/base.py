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
