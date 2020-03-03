import numpy as np

from alphazero.alphazero.mcts import MCTS
from alphazero.alphazero.nn import AlphaZeroNN
from alphazero.games.base import GameState, Move, Game
from .base import Agent


class AlphaZeroAgent(Agent):
    def __init__(self, game: Game, nn: AlphaZeroNN):
        super().__init__()
        self.game = game
        self.nn = nn
        self.mcts = MCTS(game, nn, num_simulations=7, c_puct=1.)

    def select_move(self, state: GameState) -> Move:
        policy = self.mcts.get_policy(state)
        return np.random.choice(self.game.action_space_size, p=policy)
