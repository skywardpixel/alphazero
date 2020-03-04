from alphazero.games.game import Game
from .game_state import GomokuGameState
from .move import GomokuMove
from .player import GomokuPlayer


class GomokuGame(Game[GomokuGameState, GomokuMove, GomokuPlayer]):
    def __init__(self, size: int = 15, n: int = 5):
        super().__init__()
        self.size = size
        self.n = n
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

    def reset(self) -> None:
        self._state = GomokuGameState.get_initial_state(self.size, self.n)

    def move_to_index(self, move: GomokuMove) -> int:
        return move.x * self.size + move.y

    def index_to_move(self, index: int) -> GomokuMove:
        x = index // self.size
        y = index % self.size
        return GomokuMove(x, y)
