from typing import Optional, List

from alphazero.games import GameState, Game, Player
from .board import TicTacToeBoard
from .exception import IllegalTicTacToeMoveException
from .move import TicTacToeMove
from .player import TicTacToePlayer


class TicTacToeGameState(GameState[TicTacToeMove, TicTacToePlayer]):
    def __init__(self, board: TicTacToeBoard, player: TicTacToePlayer) -> None:
        super().__init__()
        self.board = board
        self._player = player

    @property
    def player(self) -> TicTacToePlayer:
        return self._player

    @classmethod
    def get_initial_state(cls, size: int = 3) -> 'TicTacToeGameState':
        return TicTacToeGameState(TicTacToeBoard(size), TicTacToePlayer.X)

    @property
    def current_player(self) -> Player:
        return self.player

    def next(self, move: TicTacToeMove) -> 'TicTacToeGameState':
        if self.is_terminal():
            raise IllegalTicTacToeMoveException("Game has ended.")
        next_board = self.board.copy()
        next_board.apply_move(self.player, move)
        return TicTacToeGameState(next_board, self.player.opponent)

    def get_legal_moves(self) -> List[TicTacToeMove]:
        return self.board.get_legal_moves()

    def is_legal_move(self, move: TicTacToeMove) -> bool:
        return self.board.is_legal_move(move)

    def winner(self) -> Optional[TicTacToePlayer]:
        if self.board.has_won(TicTacToePlayer.X):
            return TicTacToePlayer.X
        elif self.board.has_won(TicTacToePlayer.O):
            return TicTacToePlayer.O
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
        return TicTacToeGameState(self.board.copy(), self.player.opponent)


class TicTacToeGame(Game[TicTacToeGameState, TicTacToeMove, TicTacToePlayer]):
    def __init__(self, size: int = 3):
        super().__init__()
        self.size = size
        self._state = TicTacToeGameState.get_initial_state(size)

    @property
    def state(self) -> TicTacToeGameState:
        return self._state

    def play(self, move: TicTacToeMove):
        self._state = self.state.next(move)

    def show_board(self) -> None:
        print(self.state.board)

    @property
    def is_over(self) -> bool:
        return self.state.is_terminal()

    @property
    def current_player(self) -> TicTacToePlayer:
        return self._state.player

    @property
    def winner(self) -> TicTacToePlayer:
        return self.state.winner()

    @property
    def action_space_size(self) -> int:
        return self.size * self.size

    def move_to_index(self, move: TicTacToeMove) -> int:
        return move.x * self.size + move.y

    def index_to_move(self, index: int) -> TicTacToeMove:
        x = index // self.size
        y = index % self.size
        return TicTacToeMove(x, y)
