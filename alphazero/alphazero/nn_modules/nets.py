from typing import Any, Dict

from alphazero.alphazero.nn_modules import AlphaZeroNeuralNet
from alphazero.alphazero.nn_modules.encoders import SimpleConvNetEncoder, LinearEncoder
from alphazero.alphazero.nn_modules.encoders.dual_resnet import DualResNetEncoder
from alphazero.alphazero.nn_modules.encoders.resnet import ResNetEncoder
from alphazero.alphazero.nn_modules.policy_heads \
    import LinearPolicyHead, SimpleFullyConnectedPolicyHead
from alphazero.alphazero.nn_modules.policy_heads.conv_linear import ConvLinearPolicyHead
from alphazero.alphazero.nn_modules.value_heads \
    import LinearValueHead, SimpleFullyConnectedValueHead
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


def resnet(game: Game, config: Dict[str, Any]):
    encoder = ResNetEncoder(in_channels=config['num_history'],
                            num_filters=config['num_resnet_filters'],
                            num_blocks=config['num_res_blocks'])
    policy_head = ConvLinearPolicyHead(game_size=config['game_size'],
                                       in_channels=config['num_resnet_filters'],
                                       output_dim=game.action_space_size)
    value_head = ConvLinearValueHead(game_size=config['game_size'],
                                     in_channels=config['num_resnet_filters'],
                                     hidden_dim=config['value_head_hidden_dim'])
    return AlphaZeroNeuralNet(encoder, policy_head, value_head, config)


def minimal_net(game: Game, config: Dict[str, Any]):
    encoder = LinearEncoder(config)
    policy_head = LinearPolicyHead(game.action_space_size, config)
    value_head = LinearValueHead(config)
    return AlphaZeroNeuralNet(encoder, policy_head, value_head, config)


def simple_conv_fc_net(game: Game, config: Dict[str, Any]):
    encoder = SimpleConvNetEncoder(config)
    policy_head = SimpleFullyConnectedPolicyHead(game.action_space_size, config)
    value_head = SimpleFullyConnectedValueHead(config)
    return AlphaZeroNeuralNet(encoder, policy_head, value_head, config)
