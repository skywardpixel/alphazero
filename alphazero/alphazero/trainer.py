import copy
import logging
import os
import random
from typing import List, Any, Dict, Tuple

import numpy as np
from torch import nn

from alphazero.agents.alphazero import AlphaZeroSampleAgent
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
        self.nn_trainer = NeuralNetTrainer(self.mcts.nn, self.config)

    def train(self):
        self.mcts.nn.eval()
        nn_update_history = [True] * self.config['patience']
        self._save_nn(self.mcts.nn, 'best.pth')
        train_examples_history: List[List[TrainExample]] = []
        for i in range(self.config['num_iters']):
            logger.info('BEGIN Iteration %d/%d', i + 1, self.config['num_iters'])
            examples_iter: List[TrainExample] = []
            for ep in range(self.config['num_episodes']):
                examples_ep = self.run_episode(i + 1, ep + 1)
                examples_iter.extend(examples_ep)
            logger.info('Finished collecting data, training NN')

            train_examples_history.append(examples_iter)
            if len(train_examples_history) > self.config['train_example_num_iters']:
                train_examples_history.pop(0)

            train_examples = []
            for examples_iter in train_examples_history:
                for example in examples_iter:
                    train_examples.append(example)

            random.shuffle(train_examples)
            logger.info('Training NN with %d examples', len(train_examples))

            updated = self._update_nn(train_examples, i + 1, compare=self.config['compare_nn'])
            nn_update_history.append(updated)

            if not any(nn_update_history[-self.config['patience']:]):
                # end training early if NN is not updated for over patience iterations
                logger.warning('NN not updated for %d iters, stopping early',
                               self.config['patience'])
                break

            logger.info('END Iteration %d/%d', i + 1, self.config['num_iters'])

    def run_episode(self, iteration: int, episode: int) -> List[TrainExample]:
        """
        Run an episode (full game of self-play) from the initial state
        and collect train examples throughout the game.
        """
        self.game.reset()
        self.mcts.reset()
        temp_examples = []  # (s_t, pi_t, player) triples

        game_step = 0
        while not self.game.is_over:
            canonical_state = self.game.state.canonical()
            temperature = 1 if game_step < self.config['temperature_threshold'] else 0
            pi = self.mcts.get_policy(canonical_state, temperature)
            temp_examples.append((canonical_state, pi, self.game.current_player))
            move = self.game.index_to_move(np.random.choice(self.game.action_space_size, p=pi))
            self.game.play(move)
            game_step += 1

        if self.game.winner is None:
            logger.info('Iter %2d Ep %2d - Result: tie', iteration, episode)
        else:
            logger.info('Iter %2d Ep %2d - Result: %s wins',
                        iteration, episode, self.game.winner)

        # transform to tensors for nn training
        # pylint: disable=not-callable
        examples = []
        for s, pi, player in temp_examples:
            encoded_state = self.state_encoder.encode(s)
            if self.game.winner is None:
                z = 0.
            else:
                z = +1. if player == self.game.winner else -1.
            examples.append((encoded_state,
                             torch.tensor(pi).float(),
                             z))

        examples = self._augment_examples(examples)
        return examples

    def _augment_examples(self, examples: List[TrainExample]) -> List[TrainExample]:
        augmented = []
        for s, pi, z in examples:
            augmented.extend([(ss, spi, z) for ss, spi in self.game.symmetries(s, pi)])
        return augmented

    def _update_nn(self,
                   train_data: List[TrainExample],
                   iteration: int,
                   compare: bool = True) -> bool:
        """
        Train the current NN on train_data, and updates it to
        the new one if compare is true, or the new NN defeats
        the old NN frequently enough.

        Return true iff the NN is updated.
        """
        if not compare:
            self.nn_trainer.train(train_data, iteration)
            self._save_nn(self.mcts.nn, f'iter{iteration}.pth')
            logger.info('NN training finished')
            return True

        # save old NN
        self._save_nn(self.mcts.nn, 'temp.pth')

        # train self.mcts.nn
        self.nn_trainer.train(train_data, iteration)
        logger.info('NN training finished, now pitting old NN and new NN')

        nn_old = copy.deepcopy(self.mcts.nn)
        self._load_nn(nn_old, 'temp.pth')

        old_wins, new_wins, ties = self._compare_nn(nn_old, self.mcts.nn, iteration)

        logger.info('Result: old_wins: %d, new_wins: %d, ties: %d, total: %d',
                    old_wins, new_wins, ties, self.config['nn_update_num_games'])

        update_threshold = self.config['nn_update_threshold']
        if old_wins + new_wins != 0 \
                and new_wins / (old_wins + new_wins) >= update_threshold:
            logger.info('Switching to new NN')
            self._save_nn(self.mcts.nn, 'best.pth')
            self._save_nn(self.mcts.nn, f'iter{iteration}.pth')
            return True
        else:
            logger.info('Keeping old NN')
            self._load_nn(self.mcts.nn, 'temp.pth')
            self._save_nn(self.mcts.nn, f'iter{iteration}.pth')
            return False

    def _compare_nn(self,
                    nn_old: nn.Module,
                    nn_new: nn.Module,
                    iteration: int) -> Tuple[int, int, int]:
        """
        Let nn_old and nn_new compete, and returns the results of the games.
        """
        nn_old.eval()
        nn_new.eval()
        old_wins, new_wins, ties = 0, 0, 0
        num_games = self.config['nn_update_num_games']
        for g in range(num_games):
            self.game.reset()
            agent_old = AlphaZeroSampleAgent(self.game, self.state_encoder, nn_old, self.config)
            agent_new = AlphaZeroSampleAgent(self.game, self.state_encoder, nn_new, self.config)

            # Let old start first for even-numbered games, new for odd-numbered games
            current_player = agent_old if g % 2 == 0 else agent_new

            while not self.game.is_over:
                move = current_player.select_move(self.game.state)
                self.game.play(move)
                current_player = agent_old if current_player == agent_new else agent_new

            if self.game.winner is None:
                logger.info('Iter %2d Game %2d/%2d - Result: tie',
                            iteration, g + 1, self.config['nn_update_num_games'])
                ties += 1
            else:
                if self.game.winner == self.game.canonical_player:
                    # first player wins
                    winner = 'old' if g % 2 == 0 else 'new'
                else:
                    winner = 'new' if g % 2 == 0 else 'old'
                logger.info('Iter %2d Game %2d/%2d - Result: %s (%s) wins',
                            iteration, g + 1, num_games, self.game.winner, winner)
                if winner == 'old':
                    old_wins += 1
                else:
                    new_wins += 1
        return old_wins, new_wins, ties

    def _save_nn(self, net: nn.Module, filename: str) -> None:
        torch.save(net.state_dict(), os.path.join(self.config['log_dir'], filename))

    def _load_nn(self, net: nn.Module, filename: str) -> None:
        net.load_state_dict(torch.load(os.path.join(self.config['log_dir'], filename)))
