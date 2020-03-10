from typing import Any, Dict

import numpy as np
import torch

from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.state_encoders import GameStateEncoder
from alphazero.games.game import Game
from alphazero.games.game_state import GameState
from alphazero.games.move import Move
from .base import Agent


class AlphaZeroArgMaxAgent(Agent):
    def __init__(self,
                 game: Game,
                 state_encoder: GameStateEncoder,
                 nn: torch.nn.Module,
                 config: Dict[str, Any]):
        super().__init__()
        self.game = game
        self.mcts = MonteCarloTreeSearch(game=game,
                                         state_encoder=state_encoder,
                                         nn=nn,
                                         config=config)

    def select_move(self, state: GameState) -> Move:
        policy = self.mcts.get_policy(state, temperature=0)
        return self.game.index_to_move(np.argmax(policy).item())


class AlphaZeroSampleAgent(Agent):
    def __init__(self,
                 game: Game,
                 state_encoder: GameStateEncoder,
                 nn: torch.nn.Module,
                 config: Dict[str, Any]):
        super().__init__()
        self.game = game
        self.mcts = MonteCarloTreeSearch(game=game,
                                         state_encoder=state_encoder,
                                         nn=nn,
                                         config=config)

    def select_move(self, state: GameState) -> Move:
        policy = self.mcts.get_policy(state, temperature=0)
        move_index = np.random.choice(self.game.action_space_size, p=policy)
        return self.game.index_to_move(move_index)
