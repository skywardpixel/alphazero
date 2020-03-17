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

NUM_FILTERS = [16, 32, 64, 128, 256]
NUM_GAMES = 100

setup_logger('experiment_logs', 'ttt_comp_random.log')
logger = logging.getLogger(__name__)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
logger.info('Training on %s', device)
game = TicTacToeGame()
state_encoder = TicTacToeStateEncoder(device)

config = {
    'num_simulations': 25,
    'c_puct': 1.,
    'game_size': 3,
    'num_history': 1,
    'num_res_blocks': 4,
    'device': device
}

random_agent = RandomPlayAgent()


def get_agent(num_filters: int) -> AlphaZeroArgMaxAgent:
    path_to_weights = os.path.join('pretrained', f'ttt_4blocks_{num_filters}filters.pth')
    config['num_resnet_filters'] = num_filters
    config['value_head_hidden_dim'] = num_filters
    net = dual_resnet(game, config)
    net.load_state_dict(torch.load(path_to_weights))
    return AlphaZeroArgMaxAgent(game, state_encoder, net, config)


win_rates_no_tie = []
win_rates = []


if __name__ == '__main__':
    for num_filters in NUM_FILTERS:
        agent_name = f'{num_filters}filters'
        logger.info('########## COMPARING %s TO RANDOM ##########', agent_name)
        agent = get_agent(num_filters)
        random_win, agent_win, tie = pit(game, NUM_GAMES, random_agent, agent, 'random', agent_name)
        logger.info('Win: %d, Lose: %d, Tie: %d', agent_win, random_win, tie)
        if random_win + agent_win == 0:
            logger.info('W/(W+L): N/A')
            win_rates_no_tie.append((num_filters, -1))
        else:
            win_rate_no_tie = agent_win / (agent_win + random_win)
            logger.info('W/(W+L): %.2f', win_rate_no_tie)
            win_rates_no_tie.append((num_filters, win_rate_no_tie))
        win_rate = agent_win / NUM_GAMES
        logger.info('W/all: %.2f', win_rate)
        win_rates.append((num_filters, win_rate))

print('W/W+L')
print(win_rates_no_tie)
print('W/all')
print(win_rates)
