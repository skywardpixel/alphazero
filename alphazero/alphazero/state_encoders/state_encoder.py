from typing import TypeVar, Generic

import torch

from alphazero.games import GameState

S = TypeVar('S', bound=GameState)


class GameStateEncoder(Generic[S]):
    """
    A GameStateEncoder encodes a GameState into a torch.Tensor.
    """

    def __init__(self):
        pass

    def encode(self, state: S) -> torch.Tensor:
        raise NotImplementedError
