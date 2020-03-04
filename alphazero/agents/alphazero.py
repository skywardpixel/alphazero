import numpy as np
import torch

from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.games.game import Game
from alphazero.games.game_state import GameState
from alphazero.games.move import Move
from .base import Agent


class AlphaZeroAgent(Agent):
    def __init__(self, game: Game, neural_net: torch.nn.Module):
        super().__init__()
        self.game = game
        self.neural_net = neural_net
        self.mcts = MonteCarloTreeSearch(game, self.neural_net, num_simulations=7, c_puct=1.)

    def select_move(self, state: GameState) -> Move:
        policy = self.mcts.get_policy(state)
        return np.random.choice(self.game.action_space_size, p=policy)
