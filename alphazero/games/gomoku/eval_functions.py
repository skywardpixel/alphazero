from alphazero.games.tictactoe import TicTacToeGameState, TicTacToePlayer


def simple_eval_func(state: TicTacToeGameState, player: TicTacToePlayer) -> float:
    if state.current_player != player:
        state = state.reverse_player()
    if state.is_win():
        return 99999.
    elif state.is_lose():
        return -99999.
    else:
        return 0.
