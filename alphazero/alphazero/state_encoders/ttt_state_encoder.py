from typing import Any, Dict

import torch

from alphazero.alphazero.state_encoders.state_encoder import GameStateEncoder
from alphazero.games.tictactoe import TicTacToeGameState, TicTacToePlayer
from alphazero.games.tictactoe.board import TicTacToeBoard


class TicTacToeStateEncoder(GameStateEncoder[TicTacToeGameState]):
    def __init__(self, config: Dict[str, Any]) -> None:
        super().__init__()
        self.device = config['device']

    def encode(self, state: TicTacToeGameState) -> torch.Tensor:
        history = [_board_to_matrix(state.board)]
        return torch.stack(history, dim=0).to(self.device)


def _board_to_matrix(board: TicTacToeBoard) -> torch.Tensor:
    board_matrix = torch.zeros([board.size, board.size])
    for r in range(board.size):
        for c in range(board.size):
            player = board.get(r, c)
            if player == TicTacToePlayer.X:
                board_matrix[r][c] = 1
            elif player == TicTacToePlayer.O:
                board_matrix[r][c] = -1
            else:
                board_matrix[r][c] = 0
    return board_matrix
