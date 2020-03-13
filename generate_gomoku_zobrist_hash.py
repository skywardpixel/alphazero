import random

from alphazero.games.gomoku.player import GomokuPlayer
from alphazero.games.gomoku.move import GomokuMove

SIZE = 7


def to_python(player_state):
    if player_state is None:
        return 'None'
    if player_state == GomokuPlayer.BLACK:
        return 'GomokuPlayer.BLACK'
    return 'GomokuPlayer.WHITE'


MAX63 = 0x7fffffffffffffff

table = {}
empty_board = 0
for row in range(SIZE):
    for col in range(SIZE):
        for state in (GomokuPlayer.BLACK, GomokuPlayer.WHITE):
            code = random.randint(0, MAX63)
            table[GomokuMove(row, col), state] = code

print('from .player import GomokuPlayer')
print('from .move import GomokuMove')
print()
print('HASH_CODE = {')
for (point, state), hash_code in table.items():
    print(f'    ({point!r}, {to_python(state)}): {hash_code!r},')
print('}')
print()
print(f'EMPTY_BOARD = {empty_board}')
