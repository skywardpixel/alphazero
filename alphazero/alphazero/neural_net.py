from typing import List

from torch import nn

from alphazero.alphazero.types import TrainExample


class NeuralNetTrainer:
    def __init__(self):
        pass

    def train(self, network: nn.Module, data: List[TrainExample]):
        pass
