import random
from typing import Callable

from agents.base import Agent
from games.base import Player, GameState, Move


class AlphaBetaAgent(Agent):
    def __init__(self, player: Player, depth: int, eval_fn: Callable[[GameState, Player], float]) -> None:
        self.player = player
        self.depth = depth
        self.eval_fn = eval_fn

    def select_move(self, state: GameState) -> Move:
        best_value, best_move = float('-inf'), None
        alpha, beta = float('-inf'), float('+inf')
        legal_moves = state.get_legal_moves()
        random.shuffle(legal_moves)
        for move in legal_moves:
            next_state = state.next(move)
            value = self._value(next_state, self.depth - 1, alpha, beta)
            if value > best_value:
                best_move = move
                best_value = value
            alpha = max(alpha, value)
        return best_move

    def _value(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        if state.is_terminal() or depth == 0:
            return self.eval_fn(state, self.player)
        if state.current_player == self.player:
            return self._max_value(state, depth, alpha, beta)
        else:
            return self._min_value(state, depth, alpha, beta)

    def _max_value(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        v = float('-inf')
        legal_moves = state.get_legal_moves()
        random.shuffle(legal_moves)
        for move in legal_moves:
            next_state = state.next(move)
            branch_value = self._value(next_state, depth - 1, alpha, beta)
            v = max(v, branch_value)
            if v > beta:
                return v
            alpha = max(alpha, v)
        return v

    def _min_value(self, state: GameState, depth: int, alpha: float, beta: float) -> float:
        v = float('+inf')
        legal_moves = state.get_legal_moves()
        random.shuffle(legal_moves)
        for move in legal_moves:
            next_state = state.next(move)
            branch_value = self._value(next_state, depth - 1, alpha, beta)
            v = min(v, branch_value)
            if v < alpha:
                return v
            beta = min(beta, v)
        return v