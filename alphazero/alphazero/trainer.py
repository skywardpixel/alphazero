import copy
import logging
from typing import List, Any, Dict, Tuple

import numpy as np
from torch import nn

from alphazero.agents.alphazero import AlphaZeroArgMaxAgent
from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.types import TrainExample
from alphazero.games import Game
from .nn_trainer import NeuralNetTrainer
from .state_encoders import GameStateEncoder, torch

logger = logging.getLogger(__name__)


class AlphaZeroTrainer:
    def __init__(self,
                 game: Game,
                 state_encoder: GameStateEncoder,
                 mcts: MonteCarloTreeSearch,
                 config: Dict[str, Any]):
        self.game = game
        self.state_encoder = state_encoder
        self.mcts = mcts
        self.config = config

    def train(self):
        # pylint: disable=too-many-locals
        self.mcts.nn.eval()
        patience = self.config['patience']
        log_dir = self.config['log_dir']
        nn_updated = [True] * patience
        torch.save(self.mcts.nn.state_dict(), f'{log_dir}/best.pth')
        for i in range(self.config['num_iters']):
            logger.info('Iteration %d/%d', i + 1, self.config['num_iters'])
            examples_iter: List[TrainExample] = []
            for ep in range(self.config['num_episodes']):
                logger.info('Episode %d/%d', ep + 1, self.config['num_episodes'])
                self.mcts.reset()
                examples_ep = self.run_episode()
                examples_iter.extend(examples_ep)

            logger.info('Finished collecting data, training NN')
            np.random.shuffle(examples_iter)

            nn_old = copy.deepcopy(self.mcts.nn)
            nn_trainer = NeuralNetTrainer(self.mcts.nn, self.config)
            nn_trainer.train(examples_iter, i + 1)
            torch.save(self.mcts.nn.state_dict(), f'{log_dir}/iter{i}.pth')

            logger.info('NN training finished, now pitting old NN and new NN')
            nn_new = self.mcts.nn

            pit_num_games = self.config['nn_update_num_games']
            old_wins, new_wins, ties = self._compare_nn(nn_old, nn_new, pit_num_games)
            logger.info('Result: old_wins: %d, new_wins: %d, ties: %d, total: %d',
                        old_wins, new_wins, ties, pit_num_games)
            new_win_rate = new_wins / pit_num_games
            if new_win_rate > self.config['nn_update_threshold']:
                logger.info('switching to new NN')
                torch.save(self.mcts.nn.state_dict(), f'{log_dir}/best.pth')
                nn_updated.append(True)
            else:
                logger.info('keeping old NN')
                self.mcts.nn = nn_old
                nn_updated.append(False)

            if all(not x for x in nn_updated[-patience:]):
                # end training early if NN is not updated for over patience iterations
                logger.info('NN not updated for %d iters, stopping early', patience)
                break

    def run_episode(self) -> List[TrainExample]:
        """
        Run an episode (full game of self-play) from the initial state
        and collect train examples throughout the game.
        :return: training examples as a list of (s_t, pi_t, z_t) triples
        """
        self.game.reset()
        examples: List[TrainExample] = []  # (s_t, pi_t, z_t) tuples

        while not self.game.is_over:
            pi = self.mcts.get_policy(self.game.state)
            # pylint: disable=fixme
            # TODO: augmentation by symmetries, what about history?
            examples.append((self.game.state.canonical(), pi, None))
            move_index = np.random.choice(self.game.action_space_size, p=pi)
            move = self.game.index_to_move(move_index)
            self.game.play(move)

        if self.game.winner is None:
            logger.info('Result: tie')
        else:
            logger.info('Result: %s wins', self.game.winner)

        if self.game.winner is None:
            z = 0.
        else:
            z = +1. if self.game.canonical_player == self.game.winner else -1.

        # transform to tensors for nn training
        # pylint: disable=not-callable
        examples = [(self.state_encoder.encode(s), torch.tensor(pi).float(), z)
                    for s, pi, _ in examples]
        return examples

    def _compare_nn(self,
                    nn_old: nn.Module,
                    nn_new: nn.Module,
                    num_games: int) -> Tuple[int, int, int]:
        """
        Let nn_old and nn_new compete, and returns True iff nn_new wins
        more than nn_update_threshold of the time.
        :param nn_old:
        :param nn_new:
        :return: a tuple (old_wins, new_wins, ties)
        """
        nn_old.eval()
        nn_new.eval()
        old_wins, new_wins, ties = 0, 0, 0
        for g in range(num_games):
            logger.info('Game %d/%d', g + 1, num_games)
            self.game.reset()
            agent_old = AlphaZeroArgMaxAgent(self.game, self.state_encoder, nn_old, self.config)
            agent_new = AlphaZeroArgMaxAgent(self.game, self.state_encoder, nn_new, self.config)

            # Let old start first for even-numbered games, new for odd-numbered games
            current_player = agent_old if g % 2 == 0 else agent_new

            while not self.game.is_over:
                move = current_player.select_move(self.game.state)
                self.game.play(move)
                current_player = agent_old if current_player == agent_new else agent_new

            if self.game.winner is None:
                logger.info('Result: tie')
                ties += 1
            else:
                if self.game.winner == self.game.canonical_player:
                    # first player wins
                    winner = 'old' if g % 2 == 0 else 'new'
                else:
                    winner = 'new' if g % 2 == 0 else 'old'
                logger.info('Result: %s (%s) wins', self.game.winner, winner)
                if winner == 'old':
                    old_wins += 1
                else:
                    new_wins += 1
        return old_wins, new_wins, ties
