from typing import List, Optional


class Player:
    pass


class Move:
    pass


class Game:
    def play(self, move: Move):
        raise NotImplementedError

    @property
    def is_terminal(self) -> bool:
        raise NotImplementedError

    def show_board(self) -> None:
        raise NotImplementedError

    @property
    def next_player(self) -> Player:
        raise NotImplementedError

    @property
    def winner(self) -> Player:
        raise NotImplementedError


class GameState:
    def __init__(self, next_player: Player) -> None:
        raise NotImplementedError

    def play(self, move: Move) -> None:
        raise NotImplementedError

    def get_legal_moves(self) -> List[Move]:
        raise NotImplementedError

    def is_terminal(self) -> bool:
        raise NotImplementedError

    def winner(self) -> Optional[Player]:
        raise NotImplementedError


class IllegalMoveException(Exception):
    pass
