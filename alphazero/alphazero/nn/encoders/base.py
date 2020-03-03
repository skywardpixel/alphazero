from typing import Any

from torch.nn import Module


class GameStateEncoder(Module):
    def forward(self, *input: Any) -> Any:  # pylint: disable=redefined-builtin
        raise NotImplementedError
