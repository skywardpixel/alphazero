from typing import Callable

from alphazero.agents.base import Agent
from alphazero.games import GameState, Move, TypeVar, Player, Generic

S = TypeVar('S', bound=GameState)
M = TypeVar('M', bound=Move)
P = TypeVar('P', bound=Player)


def cmd_line_move_reader(state: GameState, move_from_str: Callable[[str], M]):
    print('Legal moves:', state.get_legal_moves())
    move_str = input('Enter move:')
    return move_from_str(move_str)


class HumanPlayerAgent(Agent, Generic[S, M, P]):
    def __init__(self, move_reader: Callable[[S], M]):
        super().__init__()
        self.move_reader = move_reader

    def select_move(self, state: S) -> M:
        return self.move_reader(S)
