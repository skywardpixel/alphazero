import logging
from typing import Tuple

from alphazero.agents.base import Agent
from alphazero.games import Game

logger = logging.getLogger(__name__)


def pit(game: Game, num_games: int, agent1: Agent, agent2: Agent,
        agent1_name: str = 'agent1', agent2_name: str = 'agent2') -> Tuple[int, int, int]:
    agent1_wins, agent2_wins, ties = 0, 0, 0
    for g in range(num_games):
        game.reset()
        agent1.reset()
        agent2.reset()

        # Let old start first for even-numbered games, new for odd-numbered games
        current_player = agent1 if g % 2 == 0 else agent2

        while not game.is_over:
            move = current_player.select_move(game.state)
            game.play(move)
            current_player = agent2 if current_player == agent1 else agent1

        if game.winner is None:
            logger.info('Game %2d/%2d - Result: tie', g + 1, num_games)
            ties += 1
        else:
            if game.winner == game.canonical_player:
                # first player wins
                winner = agent1_name if g % 2 == 0 else agent2_name
            else:
                winner = agent2_name if g % 2 == 0 else agent1_name
            logger.info('Game %2d/%2d - Result: %s (%s) wins',
                        g + 1, num_games, game.winner, winner)
            if winner == agent1_name:
                agent1_wins += 1
            else:
                agent2_wins += 1
    return agent1_wins, agent2_wins, ties
