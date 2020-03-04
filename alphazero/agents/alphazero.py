import numpy as np
import torch

from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.state_encoders import GameStateEncoder
from alphazero.games.game import Game
from alphazero.games.game_state import GameState
from alphazero.games.move import Move
from .base import Agent


class AlphaZeroAgent(Agent):
    def __init__(self,
                 game: Game,
                 state_encoder: GameStateEncoder,
                 nn: torch.nn.Module,
                 num_simulations: int,
                 c_puct: float):
        super().__init__()
        self.game = game
        self.state_encoder = state_encoder
        self.mcts = MonteCarloTreeSearch(game=game,
                                         state_encoder=state_encoder,
                                         nn=nn,
                                         num_simulations=num_simulations,
                                         c_puct=c_puct)

    def select_move(self, state: GameState) -> Move:
        policy = self.mcts.get_policy(state)
        return np.random.choice(self.game.action_space_size, p=policy)
