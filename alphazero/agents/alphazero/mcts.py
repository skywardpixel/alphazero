from typing import List

from alphazero.games.base import GameState, Move


class MCTSNode:
    def __init__(self,
                 game_state: GameState,
                 parent: 'MCTSNode' = None,
                 move: Move = None) -> None:
        self.game_state: GameState = game_state
        self.parent: 'MCTSNode' = parent
        self.move: Move = move
        self.children: List['MCTSNode'] = []
        self.actions_to_expand: List[Move] = self.game_state.get_legal_moves()

    def randomly_expand(self):
        pass
