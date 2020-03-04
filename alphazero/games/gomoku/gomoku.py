from typing import Optional, List

from alphazero.games.game import Game
from alphazero.games.game_state import GameState
from alphazero.games.player import Player
from .board import GomokuBoard
from .exception import IllegalGomokuMoveException
from .move import GomokuMove
from .player import GomokuPlayer


class GomokuGameState(GameState[GomokuMove, GomokuPlayer]):
    def __init__(self, board: GomokuBoard, player: GomokuPlayer) -> None:
        super().__init__()
        self.board = board
        self._player = player

    @property
    def player(self) -> GomokuPlayer:
        return self._player

    @classmethod
    def get_initial_state(cls, size: int = 15, n: int = 5):
        return GomokuGameState(GomokuBoard(size, n), GomokuPlayer.BLACK)

    @property
    def current_player(self) -> Player:
        return self.player

    def next(self, move: GomokuMove) -> 'GomokuGameState':
        if self.is_terminal():
            raise IllegalGomokuMoveException("Game has ended.")
        next_board = self.board.copy()
        next_board.apply_move(self.player, move)
        return GomokuGameState(next_board, self.player.opponent)

    def get_legal_moves(self) -> List[GomokuMove]:
        return self.board.get_legal_moves(within=2)

    def is_legal_move(self, move: GomokuMove) -> bool:
        return self.board.is_legal_move(move)

    def winner(self) -> Optional[GomokuPlayer]:
        if self.board.has_won(GomokuPlayer.BLACK):
            return GomokuPlayer.BLACK
        elif self.board.has_won(GomokuPlayer.WHITE):
            return GomokuPlayer.WHITE
        else:
            return None

    def is_terminal(self) -> bool:
        return self.is_win() or self.is_lose() or self.is_tie()

    def is_win(self) -> bool:
        return self.winner() == self.player

    def is_lose(self) -> bool:
        return self.winner() == self.player.opponent

    def is_tie(self) -> bool:
        return self.winner() is None and self.board.full()

    def reverse_player(self) -> 'GameState':
        # reversed_board = self.board.copy()
        # for p in self.board.grid:
        #     reversed_board.grid[p] = self.board.grid[p].opponent
        return GomokuGameState(self.board.copy(), self.player.opponent)


class GomokuGame(Game[GomokuGameState, GomokuMove, GomokuPlayer]):
    def __init__(self, size: int = 15, n: int = 5):
        super().__init__()
        self.size = size
        self._state = GomokuGameState.get_initial_state(size, n)

    @property
    def state(self) -> GomokuGameState:
        return self._state

    def play(self, move: GomokuMove):
        self._state = self.state.next(move)

    def show_board(self) -> None:
        print(self.state.board)

    @property
    def is_over(self) -> bool:
        return self.state.is_terminal()

    @property
    def current_player(self) -> GomokuPlayer:
        return self.state.player

    @property
    def winner(self) -> GomokuPlayer:
        return self.state.winner()

    @property
    def action_space_size(self) -> int:
        return self.size * self.size

    def move_to_index(self, move: GomokuMove) -> int:
        return move.x * self.size + move.y

    def index_to_move(self, index: int) -> GomokuMove:
        x = index // self.size
        y = index % self.size
        return GomokuMove(x, y)
