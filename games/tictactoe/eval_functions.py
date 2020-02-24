from games.tictactoe import TicTacToeGameState, TicTacToePlayer, TicTacToeBoard


def simple_eval_func(state: TicTacToeGameState, player: TicTacToePlayer) -> float:
    if state.current_player != player:
        state = state.reverse_player()
    if state.is_win():
        return 99999.
    elif state.is_lose():
        return -99999.
    else:
        return 0.


def better_eval_func(state: TicTacToeGameState, player: TicTacToePlayer) -> float:
    if state.is_tie():
        return 0.
    if state.current_player == player:
        if state.is_win():
            return 99999.
        elif state.is_lose():
            return -99999.
    else:
        if state.is_win():
            return -99999.
        elif state.is_lose():
            return 99999.

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
