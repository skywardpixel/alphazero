import random

from alphazero.games.tictactoe.player import TicTacToePlayer
from alphazero.games.tictactoe.move import TicTacToeMove

SIZE = 3


def to_python(player_state):
    if player_state is None:
        return 'None'
    if player_state == TicTacToePlayer.X:
        return 'TicTacToePlayer.X'
    return 'TicTacToePlayer.O'


MAX63 = 0x7fffffffffffffff

table = {}
empty_board = 0
for row in range(SIZE):
    for col in range(SIZE):
        for state in (TicTacToePlayer.X, TicTacToePlayer.O):
            code = random.randint(0, MAX63)
            table[TicTacToeMove(row, col), state] = code

print('from .player import TicTacToePlayer')
print('from .move import TicTacToeMove')
print()
print('HASH_CODE = {')
for (point, state), hash_code in table.items():
    print(f'    ({point!r}, {to_python(state)}): {hash_code!r},')
print('}')
print()
print(f'EMPTY_BOARD = {empty_board}')
