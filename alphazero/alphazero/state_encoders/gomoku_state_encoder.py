from typing import Any, Dict

import torch

from alphazero.alphazero.state_encoders.state_encoder import GameStateEncoder
from alphazero.games.gomoku import GomokuGameState, GomokuPlayer
from alphazero.games.gomoku.board import GomokuBoard


class GomokuStateEncoder(GameStateEncoder[GomokuGameState]):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.device = config['device']

    def encode(self, state: GomokuGameState) -> torch.Tensor:
        history = [_board_to_matrix(state.board)]
        return torch.stack(history, dim=0).to(self.device)


def _board_to_matrix(board: GomokuBoard) -> torch.Tensor:
    board_matrix = torch.zeros([board.size, board.size])
    for r in range(board.size):
        for c in range(board.size):
            player = board.get(r, c)
            if player == GomokuPlayer.BLACK:
                board_matrix[r][c] = 1.
            elif player == GomokuPlayer.WHITE:
                board_matrix[r][c] = -1.
            else:
                board_matrix[r][c] = 0.
    return board_matrix
