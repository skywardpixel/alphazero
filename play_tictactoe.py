from agents.minimax import MinimaxAgent, AlphaBetaAgent
from games.tictactoe import TicTacToeGame, TicTacToeMove, TicTacToePlayer, GameState, Player


def read_move(player: TicTacToePlayer) -> TicTacToeMove:
    x, y = input(f"{player.name} move: ").split()
    x, y = int(x), int(y)
    return TicTacToeMove(x, y)


def simple_eval_func(state: GameState, player: Player) -> float:
    if state.current_player != player:
        state = state.reverse_player()
    if state.is_win():
        return 99999.
    elif state.is_lose():
        return -99999.
    else:
        return 0.


game = TicTacToeGame()
agent = AlphaBetaAgent(TicTacToePlayer.O, depth=6, eval_fn=simple_eval_func)

while not game.is_over:
    game.show_board()
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
if game.winner:
    print(f"{game.winner.name} wins!")
else:
    print("tie :(")
