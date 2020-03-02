from typing import Optional, List

from alphazero.games.base import Player, GameState, Game
from .board import GoBoard
from .exception import IllegalGoMoveException
from .move import GoMove
from .player import GoPlayer
from .scoring import compute_game_result


class GoGameState(GameState):
    def __init__(self,
                 board: GoBoard,
                 player: GoPlayer,
                 previous_state: Optional['GoGameState'],
                 last_move: Optional[GoMove]) -> None:
        super().__init__()
        self.board = board
        self.player = player
        self.previous_state = previous_state
        self.last_move = last_move

    @classmethod
    def get_initial_state(cls, size: int = 9) -> 'GoGameState':
        return GoGameState(GoBoard(size), GoPlayer.BLACK, None, None)

    @property
    def current_player(self) -> Player:
        return self.player

    def next(self, move: GoMove) -> 'GoGameState':
        if self.is_terminal():
            raise IllegalGoMoveException("Game has ended.")
        next_board = self.board.copy()
        if move.is_play:
            next_board.place_stone(self.player, move.point)
        return GoGameState(next_board, self.player.opponent, self, move)

    def get_legal_moves(self) -> List[GoMove]:
        legal_play_moves = [GoMove(p)
                            for p in self.board.get_empty_points()
                            if self.is_legal_move(GoMove(p))]
        return legal_play_moves + [GoMove.resign(), GoMove.pass_turn()]

    def is_legal_move(self, move: GoMove):
        if self.is_terminal():
            return False
        if move.is_resign or move.is_pass:
            return True
        is_self_capture = self.board.is_move_self_capture(self.player, move.point)
        return self.board.get(move.x, move.y) is None and not is_self_capture

    def winner(self) -> Optional[GoPlayer]:
        if not self.is_terminal():
            return None
        if self.last_move.is_resign:
            return self.player
        game_result = compute_game_result(self)
        return game_result.winner

    def is_terminal(self) -> bool:
        if self.last_move is None:
            # initial state
            return False
        if self.last_move.is_resign:
            return True
        if self.last_move.is_pass:
            second_to_last_move = self.previous_state.last_move
            if second_to_last_move is not None and second_to_last_move.is_pass:
                return True
        return False

    def is_win(self) -> bool:
        return self.winner() == self.player

    def is_lose(self) -> bool:
        return self.winner() == self.player.opponent

    def is_tie(self) -> bool:
        return self.winner() is None and self.is_terminal()

    def reverse_player(self) -> 'GameState':
        # reversed_board = self.board.copy()
        # for p in self.board.grid:
        #     reversed_board.grid[p] = self.board.grid[p].opponent
        return GoGameState(self.board.copy(),
                           self.player.opponent,
                           self.previous_state,
                           self.last_move)


class GoGame(Game):
    def __init__(self, size: int = 9):
        super().__init__()
        self.state = GoGameState.get_initial_state(size)

    def play(self, move: GoMove):
        self.state = self.state.next(move)

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
