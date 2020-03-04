from typing import NewType, Tuple, Optional, List

import torch

State = NewType('State', int)
Policy = NewType('Policy', torch.Tensor)
Value = NewType('Value', float)

TrainExample = NewType('TrainExample', Tuple[State, List[float], Optional[Value]])
