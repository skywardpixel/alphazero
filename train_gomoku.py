import logging
import pathlib

import torch
import yaml
from torchsummary import summary

import alphazero.alphazero.nn_modules.nets as nets
from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.state_encoders.gomoku_state_encoder import GomokuStateEncoder
from alphazero.alphazero.trainer import AlphaZeroTrainer
from alphazero.games.gomoku import GomokuGame
from alphazero.util.logging_config import setup_logger

with open('gomoku.yaml', 'r') as f:
    config = yaml.safe_load(f)
pathlib.Path(config['log_dir']).mkdir(parents=True, exist_ok=True)

config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

setup_logger(config['log_dir'], 'train.log')
logger = logging.getLogger(__name__)
logger.info('** Training on %s **', config['device'])
logger.info(config)


if __name__ == '__main__':
    game = GomokuGame(config['game_size'])
    state_encoder = GomokuStateEncoder(config)

    net = getattr(nets, config['nn_arch'])(game, config)

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
