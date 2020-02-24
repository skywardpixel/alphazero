import os

from agents.minimax.alphabeta import AlphaBetaAgent
from games.tictactoe.eval_functions import simple_eval_func, better_eval_func
from games.tictactoe.tictactoe import TicTacToeGame, TicTacToeMove, TicTacToePlayer


def clear():
    os.system('cls') if os.name == 'nt' else os.system('clear')


def read_move(player: TicTacToePlayer) -> TicTacToeMove:
    x, y = input(f"{player.name} move: ").split()
    x, y = int(x), int(y)
    return TicTacToeMove(x, y)


game = TicTacToeGame()
# TODO: fix typing for eval_fn
agent = AlphaBetaAgent(TicTacToePlayer.O, depth=6, eval_fn=better_eval_func)

while not game.is_over:
    clear()
    game.show_board()
    # print(f"current state score by eval func: {agent.eval_fn(game.state, agent.player)}")
    if game.current_player == TicTacToePlayer.X:
        move = read_move(game.current_player)
        while not game.state.board.is_legal_move(move):
            print("Illegal move, try again")
            move = read_move(game.current_player)
    else:
        move = agent.select_move(game.state)
        print(f"Agent O move: {move}")
    game.play(move)

print("--- GAME OVER ---")
game.show_board()
print(f"current state score by eval func: {agent.eval_fn(game.state, agent.player)}")
if game.winner:
    print(f"{game.winner.name} wins!")
else:
    print("tie :(")
