import logging
import sys

import yaml

from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.nn_modules import AlphaZeroNeuralNet
from alphazero.alphazero.nn_modules.encoders.linear import LinearEncoder
from alphazero.alphazero.nn_modules.policy_heads.linear import LinearPolicyHead
from alphazero.alphazero.nn_modules.value_heads.linear import LinearValueHead
from alphazero.alphazero.state_encoders.go_state_encoder import GoStateEncoder
from alphazero.alphazero.trainer import AlphaZeroTrainer
from alphazero.games.go import GoGame

FORMAT = '%(asctime)s - %(name)-15s - %(levelname)s - %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=FORMAT, datefmt='%m/%d/%Y %I:%M:%S %p')

with open('go.yaml', 'r') as f:
    config = yaml.safe_load(f)

if __name__ == '__main__':
    game = GoGame(config['game_size'])
    state_encoder = GoStateEncoder(num_history=3)

    encoder = LinearEncoder(config)
    value_head = LinearValueHead(config)
    policy_head = LinearPolicyHead(game.action_space_size, config)

    net = AlphaZeroNeuralNet(encoder, policy_head, value_head)
    mcts = MonteCarloTreeSearch(game=game,
                                state_encoder=state_encoder,
                                nn=net,
                                config=config)
    trainer = AlphaZeroTrainer(game=game,
                               state_encoder=state_encoder,
                               mcts=mcts,
                               config=config)
    trainer.train()
