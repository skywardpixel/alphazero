from alphazero.games.base import GameState, Move


class Agent:
    def select_move(self, state: GameState) -> Move:
        raise NotImplementedError
