import copy
import logging
import random
from typing import List, Any, Dict, Tuple

import numpy as np
from torch import nn

from alphazero.agents.alphazero import AlphaZeroSampleAgent
from alphazero.alphazero.mcts import MonteCarloTreeSearch
from alphazero.alphazero.types import TrainExample
from alphazero.games import Game, Player
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

        train_examples_history = []
        for i in range(self.config['num_iters']):
            logger.info('Iteration %d/%d', i + 1, self.config['num_iters'])
            examples_iter: List[TrainExample] = []
            for ep in range(self.config['num_episodes']):
                self.mcts.reset()
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

            torch.save(self.mcts.nn.state_dict(), f'{log_dir}/temp.pth')
            nn_trainer = NeuralNetTrainer(self.mcts.nn, self.config)
            nn_trainer.train(train_examples, i + 1)

            logger.info('NN training finished, now pitting old NN and new NN')
            nn_new = self.mcts.nn
            nn_old = copy.deepcopy(self.mcts.nn)
            nn_old.load_state_dict(torch.load(f'{log_dir}/temp.pth'))
            nn_new.eval()
            nn_old.eval()

            pit_num_games = self.config['nn_update_num_games']
            old_wins, new_wins, ties = self._compare_nn(nn_old, nn_new, pit_num_games, i + 1)
            logger.info('Result: old_wins: %d, new_wins: %d, ties: %d, total: %d',
                        old_wins, new_wins, ties, pit_num_games)

            update_threshold = self.config['nn_update_threshold']
            if old_wins + new_wins != 0 \
                    and new_wins / (old_wins + new_wins) >= update_threshold:
                logger.info('Switching to new NN')
                torch.save(self.mcts.nn.state_dict(), f'{log_dir}/best.pth')
                nn_updated.append(True)
            else:
                logger.info('Keeping old NN')
                self.mcts.nn = nn_old
                nn_updated.append(False)

            if not any(nn_updated[-patience:]):
                # end training early if NN is not updated for over patience iterations
                logger.info('NN not updated for %d iters, stopping early', patience)
                break

    def run_episode(self, iteration: int, episode: int) -> List[TrainExample]:
        """
        Run an episode (full game of self-play) from the initial state
        and collect train examples throughout the game.
        """
        self.game.reset()
        temp_examples = []  # (s_t, pi_t, player) triples

        game_step = 0
        while not self.game.is_over:
            canonical_state = self.game.state.canonical()
            temperature = 1 if game_step < self.config['temperature_threshold'] else 0
            pi = self.mcts.get_policy(canonical_state, temperature)
            temp_examples.append((canonical_state, pi, self.game.current_player))
            move_index = np.random.choice(self.game.action_space_size, p=pi)
            move = self.game.index_to_move(move_index)
            self.game.play(move)
            game_step += 1

        if self.game.winner is None:
            logger.info('Iter %2d Ep %2d - Result: tie', iteration, episode)
        else:
            logger.info('Iter %2d Ep %2d - Result: %s wins', iteration, episode, self.game.winner)

        def compute_z(player: Player):
            if self.game.winner is None:
                z = 0.
            else:
                z = +1. if player == self.game.winner else -1.
            return z

        # transform to tensors for nn training
        # pylint: disable=not-callable

        # examples = [(self.state_encoder.encode(s), torch.tensor(pi).float(), compute_z(p))
        #             for s, pi, p in temp_examples]

        augmented_examples = []
        for s, pi, player in temp_examples:
            encoded_state = self.state_encoder.encode(s)
            for sym_state, sym_policy in self.game.symmetries(encoded_state, pi):
                augmented_examples.append((sym_state, torch.tensor(pi).float(), compute_z(player)))

        return augmented_examples

    def _compare_nn(self,
                    nn_old: nn.Module,
                    nn_new: nn.Module,
                    num_games: int,
                    iteration: int) -> Tuple[int, int, int]:
        """
        Let nn_old and nn_new compete, and returns the results of the games.
        """
        nn_old.eval()
        nn_new.eval()
        old_wins, new_wins, ties = 0, 0, 0
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
                logger.info('Iter %2d Game %d/%d - Result: tie', iteration, g + 1, num_games)
                ties += 1
            else:
                if self.game.winner == self.game.canonical_player:
                    # first player wins
                    winner = 'old' if g % 2 == 0 else 'new'
                else:
                    winner = 'new' if g % 2 == 0 else 'old'
                logger.info('Iter %2d Game %d/%d - Result: %s (%s) wins',
                            iteration, g + 1, num_games, self.game.winner, winner)
                if winner == 'old':
                    old_wins += 1
                else:
                    new_wins += 1
        return old_wins, new_wins, ties
