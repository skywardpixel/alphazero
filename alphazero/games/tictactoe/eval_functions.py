from .board import TicTacToeBoard
from .game_state import TicTacToeGameState
from .player import TicTacToePlayer


def simple_eval_func(state: TicTacToeGameState, player: TicTacToePlayer) -> float:
    state = state.canonical()
    if state.is_win():
        score = 99999.
    elif state.is_lose():
        score = -99999.
    else:
        score = 0.
    # above are scores for X
    if player == TicTacToePlayer.O:
        return -score
    return score


def better_eval_func(state: TicTacToeGameState, player: TicTacToePlayer) -> float:
    state = state.canonical()
    if state.is_terminal():
        return simple_eval_func(state, player)

    def count_good_rows(board: TicTacToeBoard, count_player: TicTacToePlayer):
        num_good_rows = 0
        for r in range(board.size):
            empty, adversary = 0, 0
            for c in range(board.size):
                if board.get(r, c) == count_player.opponent:
                    adversary += 1
                elif board.get(r, c) is None:
                    empty += 1
            if empty == 1 and adversary == 0:
                num_good_rows += 1
        return num_good_rows

    def count_good_columns(board: TicTacToeBoard, count_player: TicTacToePlayer):
        num_good_columns = 0
        for c in range(board.size):
            empty, adversary = 0, 0
            for r in range(board.size):
                if board.get(r, c) == count_player.opponent:
                    adversary += 1
                elif board.get(r, c) is None:
                    empty += 1
            if empty == 1 and adversary == 0:
                num_good_columns += 1
        return num_good_columns

    def count_good_diagonals(board: TicTacToeBoard, count_player: TicTacToePlayer):
        num_good_diagonals = 0
        # major
        empty, adversary = 0, 0
        for i in range(board.size):
            if board.get(i, i) == count_player.opponent:
                adversary += 1
            elif board.get(i, i) is None:
                empty += 1
        if empty == 1 and adversary == 0:
            num_good_diagonals += 1
        # minor
        empty, adversary = 0, 0
        for i in range(board.size):
            if board.get(board.size - 1 - i, i) == count_player.opponent:
                adversary += 1
            elif board.get(i, i) is None:
                empty += 1
        if empty == 1 and adversary == 0:
            num_good_diagonals += 1
        return num_good_diagonals

    board = state.board
    return 100. * (count_good_rows(board, player)
                   + count_good_columns(board, player)
                   + count_good_diagonals(board, player)
                   - count_good_rows(board, player.opponent)
                   - count_good_columns(board, player.opponent)
                   - count_good_diagonals(board, player.opponent))
