import pathlib

import torch
import yaml
from torchsummary import summary

from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.nn_modules.nets import dual_resnet
from alphazero.alphazero.state_encoders.ttt_state_encoder import TicTacToeStateEncoder
from alphazero.alphazero.trainer import AlphaZeroTrainer
from alphazero.games.tictactoe import TicTacToeGame
from alphazero.util.logging_config import setup_logger

with open('tictactoe.yaml', 'r') as f:
    config = yaml.safe_load(f)
pathlib.Path(config['log_dir']).mkdir(parents=True, exist_ok=True)

setup_logger(config['log_dir'], 'train.log')

config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    game = TicTacToeGame(config['game_size'])
    state_encoder = TicTacToeStateEncoder(config)

    net = dual_resnet(game, config)
    summary(net,
            input_size=(config['num_history'], config['game_size'], config['game_size']),
            batch_size=config['batch_size'])

    mcts = MonteCarloTreeSearch(game=game,
                                state_encoder=state_encoder,
                                nn=net,
                                config=config)
    trainer = AlphaZeroTrainer(game=game,
                               state_encoder=state_encoder,
                               mcts=mcts,
                               config=config)
    trainer.train()
