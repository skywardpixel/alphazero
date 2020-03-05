from typing import Tuple, Optional, List, Union

import torch

from alphazero.games import GameState, Move

State = Union[GameState, torch.Tensor, int]
Action = Union[Move, int]
Policy = Union[List[float], torch.Tensor]
Value = float

TrainExample = Tuple[State, Policy, Optional[Value]]
