import random

from alphazero.games.go.player import GoPlayer
from alphazero.games.go.point import GoPoint

SIZE = 9


def to_python(player_state):
    if player_state is None:
        return 'None'
    if player_state == GoPlayer.BLACK:
        return 'GoPlayer.BLACK'
    return 'GoPlayer.WHITE'


MAX63 = 0x7fffffffffffffff

table = {}
empty_board = 0
for row in range(SIZE):
    for col in range(SIZE):
        for state in (GoPlayer.BLACK, GoPlayer.WHITE):
            code = random.randint(0, MAX63)
            table[GoPoint(row, col), state] = code

print('from .player import GoPlayer')
print('from .point import GoPoint')
print()
print('HASH_CODE = {')
for (point, state), hash_code in table.items():
    print(f'    ({point!r}, {to_python(state)}): {hash_code!r},')
print('}')
print()
print(f'EMPTY_BOARD = {empty_board}')
