import os

from alphazero.agents.minimax import AlphaBetaAgent
from alphazero.games.gomoku import GomokuPlayer, GomokuMove, GomokuGame
from alphazero.games.gomoku.eval_functions import simple_eval_func


def clear():
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')


def read_move(player: GomokuPlayer) -> GomokuMove:
    x, y = input(f'{player.name} move: ').split()
    x, y = int(x), int(y)
    return GomokuMove(x, y)


game = GomokuGame(7, 5)
agent = AlphaBetaAgent(GomokuPlayer.WHITE, depth=2, eval_fn=simple_eval_func)

while not game.is_over:
    clear()
    game.show_board()
    # print(f"current state score by eval func: {agent.eval_fn(game.state, agent.player)}")
    if game.current_player == GomokuPlayer.BLACK:
        move = read_move(game.current_player)
        while not game.state.is_legal_move(move):
            print('Illegal move, try again')
            move = read_move(game.current_player)
    else:
        print(f'legal moves: f{game.state.get_legal_moves()}')
        move = agent.select_move(game.state)
        print(f'Agent WHITE move: {move}')
    game.play(move)

print('--- GAME OVER ---')
game.show_board()
print(f'current state score by eval func: {agent.eval_fn(game.state, agent.player)}')
if game.winner:
    print(f'{game.winner!s} wins!')
else:
    print('tie :(')
