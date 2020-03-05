from alphazero.games.game import Game
from .game_state import GoGameState
from .move import GoMove
from .player import GoPlayer


class GoGame(Game[GoGameState, GoMove, GoPlayer]):
    def __init__(self, size: int = 9):
        super().__init__()
        self.size = size
        self._state = GoGameState.get_initial_state(size)

    @property
    def state(self) -> GoGameState:
        return self._state

    def play(self, move: GoMove):
        self._state = self.state.next(move)

    def show_board(self) -> None:
        print(self.state.board)

    @property
    def is_over(self) -> bool:
        return self.state.is_terminal()

    @property
    def current_player(self) -> GoPlayer:
        return self.state.player

    @property
    def winner(self) -> GoPlayer:
        return self.state.winner()

    @property
    def action_space_size(self) -> int:
        return self.size * self.size + 2

    def reset(self) -> None:
        self._state = GoGameState.get_initial_state(self.size)

    def move_to_index(self, move: GoMove) -> int:
        if move.is_pass:
            return 0
        if move.is_resign:
            return self.size * self.size + 1
        return move.x * self.size + move.y + 1

    def index_to_move(self, index: int) -> GoMove:
        if index == 0:
            return GoMove.pass_turn()
        if index == self.size * self.size + 1:
            return GoMove.resign()
        index -= 1
        x = index // self.size
        y = index % self.size
        return GoMove.play(x, y)
