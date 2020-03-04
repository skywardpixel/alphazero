import logging
import sys

from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.nn_modules import AlphaZeroNeuralNet
from alphazero.alphazero.nn_modules.go_simple_conv import GoSimpleConvNetEncoder
from alphazero.alphazero.nn_modules.simple_fc_policy_head import SimpleFullyConnectedPolicyHead
from alphazero.alphazero.nn_modules.simple_fc_value_head import SimpleFullyConnectedValueHead
from alphazero.alphazero.state_encoders.go_state_encoder import GoStateEncoder
from alphazero.alphazero.trainer import AlphaZeroTrainer
from alphazero.games.go import GoGame

logging.basicConfig(stream=sys.stdout, level=logging.INFO)

if __name__ == '__main__':
    game = GoGame(9)
    state_encoder = GoStateEncoder(num_history=3)

    encoder = GoSimpleConvNetEncoder(board_size=game.size, in_channels=3, output_dim=128)
    value_head = SimpleFullyConnectedValueHead(input_dim=128)
    policy_head = SimpleFullyConnectedPolicyHead(128, game.action_space_size)

    net = AlphaZeroNeuralNet(encoder=encoder, policy_head=policy_head, value_head=value_head)
    mcts = MonteCarloTreeSearch(game=game,
                                state_encoder=state_encoder,
                                nn=net,
                                num_simulations=2,
                                c_puct=1.)
    trainer = AlphaZeroTrainer(game=game,
                               state_encoder=state_encoder,
                               mcts=mcts,
                               num_iter=2,
                               num_episode=2,
                               num_simulations=2,
                               c_puct=1.)

    trainer.train()
