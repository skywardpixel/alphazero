from typing import Tuple, Optional, List, Union

import torch

from alphazero.games import GameState

# TODO: use a more compact rep for State
State = Union[GameState, torch.Tensor]
Policy = Union[List[float], torch.Tensor]
Value = float

TrainExample = Tuple[State, Policy, Optional[Value]]
