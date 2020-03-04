from alphazero.games.game_state import GameState
from alphazero.games.move import Move


class Agent:
    def __init__(self):
        pass

    def select_move(self, state: GameState) -> Move:
        raise NotImplementedError
