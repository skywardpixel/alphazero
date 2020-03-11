from typing import List, Tuple

import numpy as np
import torch

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

    def symmetries(self, state: torch.Tensor, policy: np.ndarray) \
            -> List[Tuple[torch.Tensor, np.ndarray]]:
        result = []
        for k in range(4):
            rotated_state = torch.rot90(state, k, [2, 1])
            rotated_policy = np.zeros_like(policy)
            for idx, prob in enumerate(policy):
                old_move = self.index_to_move(idx)
                rotated_move = self._rotate_move(old_move, k)
                rotated_idx = self.move_to_index(rotated_move)
                rotated_policy[rotated_idx] = prob
            result.append((rotated_state, rotated_policy))
            result.append(self._flip(rotated_state, rotated_policy))
        return result

    def _flip(self, state: torch.Tensor, policy: np.ndarray) -> Tuple[torch.Tensor, np.ndarray]:
        flipped_state = torch.flip(state, [2])
        flipped_policy = np.zeros_like(policy)
        for idx, prob in enumerate(policy):
            old_move = self.index_to_move(idx)
            flipped_move = self._flip_move_y(old_move)
            flipped_idx = self.move_to_index(flipped_move)
            flipped_policy[flipped_idx] = prob
        return flipped_state, flipped_policy

    def _rotate_move(self, move: TicTacToeMove, k: int) -> TicTacToeMove:
        for _ in range(k % 4):
            move = TicTacToeMove(move.y, self.size - move.x - 1)
        return move

    def _flip_move_y(self, move: TicTacToeMove) -> TicTacToeMove:
        return TicTacToeMove(move.x, self.size - move.y - 1)
