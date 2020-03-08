from alphazero.games import Game
from .game_state import TicTacToeGameState
from .move import TicTacToeMove
from .player import TicTacToePlayer


class TicTacToeGame(Game[TicTacToeGameState, TicTacToeMove, TicTacToePlayer]):
    canonical_player = TicTacToePlayer.X

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

    def reset(self) -> None:
        self._state = TicTacToeGameState.get_initial_state(self.size)

    def move_to_index(self, move: TicTacToeMove) -> int:
        return move.x * self.size + move.y

    def index_to_move(self, index: int) -> TicTacToeMove:
        x = index // self.size
        y = index % self.size
        return TicTacToeMove(x, y)
