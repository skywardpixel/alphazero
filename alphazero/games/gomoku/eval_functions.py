from .game_state import GomokuGameState
from .player import GomokuPlayer


def simple_eval_func(state: GomokuGameState, player: GomokuPlayer) -> float:
    state = state.canonical()
    if state.is_win():
        score = 99999.
    elif state.is_lose():
        score = -99999.
    else:
        score = 0.
    # above are scores for BLACK
    if player == GomokuPlayer.WHITE:
        return -score
    return score
