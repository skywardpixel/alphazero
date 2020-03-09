import logging
import sys

import torch
import yaml

from alphazero.agents.alphazero import AlphaZeroArgMaxAgent
from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.nn_modules.nets import dual_resnet
from alphazero.alphazero.state_encoders.ttt_state_encoder import TicTacToeStateEncoder
from alphazero.games.tictactoe import TicTacToeGame, TicTacToePlayer, TicTacToeMove

FORMAT = '%(asctime)s - %(name)-15s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')

with open('tictactoe.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'


def read_move(player: TicTacToePlayer) -> TicTacToeMove:
    x, y = input(f"{player.name} move: ").split()
    x, y = int(x), int(y)
    return TicTacToeMove(x, y)


if __name__ == '__main__':
    game = TicTacToeGame(config['game_size'])
    state_encoder = TicTacToeStateEncoder(config)

    # encoder = LinearEncoder(config)
    # value_head = LinearValueHead(config)
    # policy_head = LinearPolicyHead(game.action_space_size, config)

    # encoder = SimpleConvNetEncoder(config)
    # value_head = SimpleFullyConnectedValueHead(config)
    # policy_head = SimpleFullyConnectedPolicyHead(game.action_space_size, config)
    #
    # net = AlphaZeroNeuralNet(encoder, policy_head, value_head, config)

    net = dual_resnet(game, config)
    mcts = MonteCarloTreeSearch(game=game,
                                state_encoder=state_encoder,
                                nn=net,
                                config=config)

    net.load_state_dict(torch.load('./tmp/ttt/best.pth'))
    net.eval()
    agent = AlphaZeroArgMaxAgent(game, state_encoder, net, config)

    while not game.is_over:
        game.show_board()
        # print(f"current state score by eval func: {agent.eval_fn(game.state, agent.player)}")
        if game.current_player == TicTacToePlayer.X:
            move = read_move(game.current_player)
            while not game.state.is_legal_move(move):
                print("Illegal move, try again")
                move = read_move(game.current_player)
        else:
            move = agent.select_move(game.state)
            print(f"Agent {game.current_player!s} move: {move}")
        game.play(move)

    print("--- GAME OVER ---")
    game.show_board()
    if game.winner:
        print(f"{game.winner.name} wins!")
    else:
        print("tie :(")
