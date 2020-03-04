from typing import Tuple, Optional, List

from alphazero.games import GameState

# TODO: use a more compact rep for State
State = GameState
Policy = List[float]
Value = float

TrainExample = Tuple[State, Policy, Optional[Value]]
