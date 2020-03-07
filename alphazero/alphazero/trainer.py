import copy
import logging
from typing import List, Any, Dict, Tuple

import numpy as np
from torch import nn

from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.types import TrainExample
from alphazero.games import Game
from .neural_net import NeuralNetTrainer
from .state_encoders import GameStateEncoder, torch
from ..agents.alphazero import AlphaZeroArgMaxAgent

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
        nn_updated = []
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
            nn_trainer.train(examples_iter)
            logger.info('NN training finished, now pitting old NN and new NN')
            nn_new = self.mcts.nn

            old_wins, new_wins, ties = \
                self._compare_nn(nn_old, nn_new, self.config['nn_update_num_games'])
            logger.info('old_wins: %d, new_wins: %d, ties: %d, total: %d',
                        old_wins, new_wins, ties, self.config['nn_update_num_games'])
            if new_wins > \
                    self.config['nn_update_threshold'] * self.config['nn_update_num_games']:
                logger.info('switching to new NN')
                nn_updated.append(True)
            else:
                logger.info('keeping old NN')
                self.mcts.nn = nn_old
                nn_updated.append(False)
            torch.save(self.mcts.nn.state_dict(),
                       f'./trained_{self.game.__class__.__name__}_iter{i}.pth')
            if len(nn_updated) > 10 and all(not x for x in nn_updated[-10:]):
                logger.info('NN not updated for 10 iters, stopping early')
                break
        torch.save(self.mcts.nn.state_dict(),
                   f'./trained_{self.game.__class__.__name__}_final.pth')

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
            # TODO: augmentation by symmetries
            examples.append((self.game.state, pi, None))
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
            z = +1. if self.game.state.canonical_player == self.game.winner else -1.

        # must transform to tensors for nn training
        # pylint: disable=not-callable
        examples = [(self.state_encoder.encode(s), torch.tensor(pi), z)
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
        old_wins, new_wins, ties = 0, 0, 0
        for g in range(num_games):
            logger.info('Game %d/%d', g + 1, num_games)
            self.game.reset()
            agent_old = AlphaZeroArgMaxAgent(self.game, self.state_encoder, nn_old, self.config)
            agent_new = AlphaZeroArgMaxAgent(self.game, self.state_encoder, nn_new, self.config)
            current_player = agent_old if g % 2 == 0 else agent_new
            while not self.game.is_over:
                move = current_player.select_move(self.game.state)
                self.game.play(move)
                current_player = agent_old if current_player == agent_new else agent_new

            if self.game.winner is None:
                logger.info('Result: tie')
                ties += 1
            else:
                if g % 2 == 0:
                    winner_agent = 'new' if self.game.winner.value == 1 else 'old'
                else:
                    winner_agent = 'old' if self.game.winner.value == 1 else 'new'
                logger.info('Result: %s (%s) wins', self.game.winner, winner_agent)
                if winner_agent == 'old':
                    old_wins += 1
                else:
                    new_wins += 1
        return old_wins, new_wins, ties
