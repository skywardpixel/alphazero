from typing import List, Type, TypeVar

from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.nn import AlphaZeroNeuralNet
from alphazero.alphazero.types import TrainExample
from alphazero.games.game import Game

G = TypeVar('G', bound=Game)


class AlphaZeroTrainer:
    def __init__(self, game_type: Type[G], num_iter: int, num_episode: int):
        self.game_type: Type = game_type
        self.num_iter = num_iter
        self.num_episode = num_episode
        self.mcts = MonteCarloTreeSearch()
        self.nn_old = AlphaZeroNeuralNet()
        self.nn_new = AlphaZeroNeuralNet()

    def run_episode(self) -> List[TrainExample]:
        examples: List[TrainExample] = []  # (s_t, pi_t, z_t) tuples
        game = self.game_type()

        while not game.is_over:
            pi = self.mcts.get_policy(game.state)
            examples.append(TrainExample((game.state, pi, None)))

        return examples

    def learn(self):
        for i in range(self.num_iter):
            examples_iter: List[TrainExample] = []
            for ep in range(self.num_episode):
                examples_iter.extend(self.run_episode())
