from typing import Any, Dict

import torch

from alphazero.alphazero.state_encoders.state_encoder import GameStateEncoder
from alphazero.games.gomoku import GomokuGameState, GomokuPlayer
from alphazero.games.gomoku.board import GomokuBoard


class GomokuStateEncoder(GameStateEncoder[GomokuGameState]):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.num_history = config['num_history']
        self.device = config['device']

    def encode(self, state: GomokuGameState) -> torch.Tensor:
        board_size = state.board.size
        history = []
        h = state
        for _ in range(self.num_history):
            if h is None:
                history.append(torch.zeros([board_size, board_size]))
                continue
            history.append(_board_to_matrix(h.board))
            h = h.previous_state
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
