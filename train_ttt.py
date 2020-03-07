import logging
import sys

import torch
import yaml

from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.nn_modules import AlphaZeroNeuralNet
from alphazero.alphazero.nn_modules.encoders import SimpleConvNetEncoder
from alphazero.alphazero.nn_modules.policy_heads import SimpleFullyConnectedPolicyHead
from alphazero.alphazero.nn_modules.value_heads import SimpleFullyConnectedValueHead
from alphazero.alphazero.state_encoders.ttt_state_encoder import TicTacToeStateEncoder
from alphazero.alphazero.trainer import AlphaZeroTrainer
from alphazero.games.tictactoe import TicTacToeGame

FORMAT = '%(asctime)s - %(name)-15s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stderr, level=logging.INFO,
                    format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')

with open('ttt.yaml', 'r') as f:
    config = yaml.safe_load(f)

config['device'] = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':
    game = TicTacToeGame(config['game_size'])
    state_encoder = TicTacToeStateEncoder(config)

    # encoder = LinearEncoder(config)
    # value_head = LinearValueHead(config)
    # policy_head = LinearPolicyHead(game.action_space_size, config)

    encoder = SimpleConvNetEncoder(config['game_size'],
                                   config['num_history'],
                                   config['encoding_dim'])
    value_head = SimpleFullyConnectedValueHead(config['encoding_dim'])
    policy_head = SimpleFullyConnectedPolicyHead(config['encoding_dim'],
                                                 game.action_space_size)

    net = AlphaZeroNeuralNet(encoder, policy_head, value_head, config)
    mcts = MonteCarloTreeSearch(game=game,
                                state_encoder=state_encoder,
                                nn=net,
                                config=config)
    trainer = AlphaZeroTrainer(game=game,
                               state_encoder=state_encoder,
                               mcts=mcts,
                               config=config)
    trainer.train()
