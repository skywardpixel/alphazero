import logging
import os

import torch

from alphazero.agents.alphazero import AlphaZeroArgMaxAgent
from alphazero.agents.random import RandomPlayAgent
from alphazero.alphazero.nn_modules.nets import dual_resnet, resnet
from alphazero.alphazero.state_encoders.ttt_state_encoder import TicTacToeStateEncoder
from alphazero.games.tictactoe import TicTacToeGame
from alphazero.util.logging_config import setup_logger
from alphazero.util.pit_agents import pit

MODELS = [
    'dualres_comp',
    'dualres_nocomp',
    'res_comp',
    'res_nocomp',
]
NUM_GAMES = 100

setup_logger('experiment_logs', 'ttt_comp_random.log')
logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Training on %s', device)
game = TicTacToeGame()
state_encoder = TicTacToeStateEncoder(device)

config = {
    'num_simulations': 50,
    'c_puct': 1.,
    'game_size': 3,
    'num_history': 1,
    'num_resnet_filters': 256,
    'value_head_hidden_dim': 256,
    'device': device
}

random_agent = RandomPlayAgent()


def get_agent(model_name: str) -> AlphaZeroArgMaxAgent:
    path_to_weights = os.path.join('pretrained', f'ttt_{model_name}.pth')
    if model_name.startswith('dualres'):
        config['num_res_blocks'] = 4
        net = dual_resnet(game, config)
    else:
        config['num_res_blocks'] = 8
        net = resnet(game, config)
    net.load_state_dict(torch.load(path_to_weights))
    return AlphaZeroArgMaxAgent(game, state_encoder, net, config)


if __name__ == '__main__':
    for name in MODELS:
        logger.info('########## COMPARING %s TO RANDOM ##########', name)
        agent = get_agent(name)
        random_win, agent_win, tie = pit(game, NUM_GAMES, random_agent, agent, 'random', name)
        logger.info('Win: %d, Lose: %d, Tie: %d', agent_win, random_win, tie)
        if random_win + agent_win == 0:
            logger.info('Win rate: N/A')
        else:
            win_rate = agent_win / (agent_win + random_win)
            logger.info('Win rate: %.2f', win_rate)
