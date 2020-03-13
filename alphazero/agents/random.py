import random
from typing import Generic, TypeVar

from alphazero.games import GameState, Move, Player
from .base import Agent

S = TypeVar('S', bound=GameState)
M = TypeVar('M', bound=Move)
P = TypeVar('P', bound=Player)


class RandomPlayAgent(Agent, Generic[S, M, P]):
    def select_move(self, state: S) -> M:
        return random.choice(state.get_legal_moves())
