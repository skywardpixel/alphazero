import os

from alphazero.games.go import GoPlayer, GoMove, GoGame


def clear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


def read_move(player: GoPlayer) -> GoMove:
    move_str = input(f'{player.name} move: ').strip().lower()
    if move_str == 'resign':
        return GoMove.resign()
    elif move_str == 'pass':
        return GoMove.pass_turn()
    x, y = move_str.split()
    x, y = int(x), int(y)
    return GoMove.play(x, y)


game = GoGame(9)

while not game.is_over:
    clear()
    game.show_board()
    # print(f"current state score by eval func: {agent.eval_fn(game.state, agent.player)}")
    move = read_move(game.current_player)
    while not game.state.is_legal_move(move):
        print('Illegal move, try again')
        move = read_move(game.current_player)
    game.play(move)

print('--- GAME OVER ---')
game.show_board()
if game.winner:
    print(f'{game.winner!s} wins!')
else:
    print('tie :(')
