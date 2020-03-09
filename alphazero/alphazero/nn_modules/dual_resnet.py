from typing import Any, Dict

from alphazero.alphazero.nn_modules import AlphaZeroNeuralNet
from alphazero.alphazero.nn_modules.encoders.dual_res import DualResNetEncoder
from alphazero.alphazero.nn_modules.policy_heads.conv_linear import ConvLinearPolicyHead
from alphazero.alphazero.nn_modules.value_heads.conv_linear import ConvLinearValueHead
from alphazero.games import Game


def dual_resnet(game: Game, config: Dict[str, Any]):
    encoder = DualResNetEncoder(in_channels=config['num_history'],
                                num_filters=config['num_resnet_filters'],
                                num_blocks=config['num_res_blocks'])
    policy_head = ConvLinearPolicyHead(game_size=config['game_size'],
                                       in_channels=config['num_resnet_filters'],
                                       output_dim=game.action_space_size)
    value_head = ConvLinearValueHead(game_size=config['game_size'],
                                     in_channels=config['num_resnet_filters'],
                                     hidden_dim=config['value_head_hidden_dim'])
    return AlphaZeroNeuralNet(encoder, policy_head, value_head, config)
