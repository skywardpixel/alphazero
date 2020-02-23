from games.ttt import TicTacToeGame, TicTacToeMove


def read_move():
    x, y = input().split()
    return int(x), int(y)

game = TicTacToeGame()

while not game.is_terminal:
    game.show_board()
    print("Legal moves", game.state.board.get_legal_moves())
    print(f"{game.next_player.name} move: ")
    x, y = read_move()
    move = TicTacToeMove(x, y)
    while not game.state.board.is_legal_move(move):
        print("Illegal move, try again")
        x, y = read_move()
        move = TicTacToeMove(x, y)
    game.play(move)

print("--- GAME OVER ---")
game.show_board()
if game.winner:
    print(f"{game.winner.name} wins!")
else:
    print("tie :(")
