from typing import Tuple, List

import torch

from alphazero.games.game import Game
from .game_state import GoGameState
from .move import GoMove
from .player import GoPlayer


class GoGame(Game[GoGameState, GoMove, GoPlayer]):
    canonical_player = GoPlayer.BLACK

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

    def symmetries(self, state: torch.Tensor, policy: torch.Tensor) \
            -> List[Tuple[torch.Tensor, torch.Tensor]]:
        result = []
        for k in range(4):
            rotated_state = torch.rot90(state, k, [2, 1])
            rotated_policy = torch.zeros_like(policy)
            for idx, prob in enumerate(policy):
                old_move = self.index_to_move(idx)
                rotated_move = self._rotate_move(old_move, k)
                rotated_idx = self.move_to_index(rotated_move)
                rotated_policy[rotated_idx] = prob
            result.append((rotated_state, rotated_policy))
            result.append(self._flip(rotated_state, rotated_policy))
        return result

    def _flip(self, state: torch.Tensor, policy: torch.Tensor) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        flipped_state = torch.flip(state, [2])
        flipped_policy = torch.zeros_like(policy)
        for idx, prob in enumerate(policy):
            old_move = self.index_to_move(idx)
            flipped_move = self._flip_move_y(old_move)
            flipped_idx = self.move_to_index(flipped_move)
            flipped_policy[flipped_idx] = prob
        return flipped_state, flipped_policy

    def _rotate_move(self, move: GoMove, k: int) -> GoMove:
        if not move.is_play:
            return move
        for _ in range(k % 4):
            move = GoMove.play(move.y, self.size - move.x - 1)
        return move

    def _flip_move_y(self, move: GoMove) -> GoMove:
        if not move.is_play:
            return move
        return GoMove.play(move.x, self.size - move.y - 1)
