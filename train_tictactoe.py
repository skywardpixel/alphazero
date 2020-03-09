import logging
import sys

import torch
import yaml

from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.nn_modules.nets import simple_conv_fc_net
from alphazero.alphazero.state_encoders.ttt_state_encoder import TicTacToeStateEncoder
from alphazero.alphazero.trainer import AlphaZeroTrainer
from alphazero.games.tictactoe import TicTacToeGame

FORMAT = '%(asctime)s - %(name)-15s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')

with open('tictactoe.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    game = TicTacToeGame(config['game_size'])
    state_encoder = TicTacToeStateEncoder(config)

    net = simple_conv_fc_net(game, config)

    mcts = MonteCarloTreeSearch(game=game,
                                state_encoder=state_encoder,
                                nn=net,
                                config=config)
    trainer = AlphaZeroTrainer(game=game,
                               state_encoder=state_encoder,
                               mcts=mcts,
                               config=config)
    trainer.train()
