from alphazero.games.base import GameState, Move


class Agent:
    def __init__(self):
        pass

    def select_move(self, state: GameState) -> Move:
        raise NotImplementedError
