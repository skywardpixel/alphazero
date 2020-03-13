from typing import List, Tuple

import torch

from alphazero.games.game import Game
from .game_state import GomokuGameState
from .move import GomokuMove
from .player import GomokuPlayer


class GomokuGame(Game[GomokuGameState, GomokuMove, GomokuPlayer]):
    canonical_player = GomokuPlayer.BLACK

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

    def _rotate_move(self, move: GomokuMove, k: int) -> GomokuMove:
        for _ in range(k % 4):
            move = GomokuMove(move.y, self.size - move.x - 1)
        return move

    def _flip_move_y(self, move: GomokuMove) -> GomokuMove:
        return GomokuMove(move.x, self.size - move.y - 1)
