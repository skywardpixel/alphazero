import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
import torch

from alphazero.alphazero.nn_modules import AlphaZeroNeuralNet
from alphazero.alphazero.state_encoders.state_encoder import GameStateEncoder
from alphazero.games.game import Game
from alphazero.games.game_state import GameState
from alphazero.games.move import Move

logger = logging.getLogger(__name__)


class MonteCarloTreeSearch:
    """
    A MCTS manages running the Monte-Carlo Tree Search algorithm
    in AlphaZero.
    """

    def __init__(self,
                 game: Game,
                 state_encoder: GameStateEncoder,
                 nn: AlphaZeroNeuralNet,
                 config: Dict[str, Any]) -> None:
        self.game = game
        self.state_encoder = state_encoder
        self.nn = nn
        self.config = config

        # Q(s, a) Q values for state-action pair
        self.Qsa: Dict[Tuple[int, int], float] = defaultdict(float)
        # N(s, a) visit counts for state-action pair
        self.Nsa: Dict[Tuple[int, int], int] = defaultdict(int)
        # N(s) visit counts for state
        self.Ns: Dict[int, int] = defaultdict(int)
        # p(s) initial policies for state, returned by NN
        self.Ps: Dict[int, np.ndarray] = dict()
        # V(s) cached valid move vector for state
        self.Vs: Dict[int, np.ndarray] = dict()

    def get_policy(self, state: GameState, temperature: float = 1) -> List[float]:
        """
        Runs the AlphaZero MCTS search from state.
        :param state: game state to search from and return the policy for
        :param temperature: temperature for output policy vector
        :return: a probability distribution on the action space of the game
        """
        state = state.canonical()
        s = state.board.zobrist_hash()
        for _ in range(self.config['num_simulations']):
            self.search(state, add_noise=True)

        counts = np.array([self.Nsa[(s, a)]
                           for a in range(self.game.action_space_size)])

        # convert count into a probability distribution
        if temperature == 0:
            # return one-hot vector for policy
            policy = np.eye(self.game.action_space_size)[np.argmax(counts)]
            return policy
        counts = counts ** (1. / temperature)
        return counts / np.sum(counts)

    def search(self, state: GameState, add_noise: bool = False) -> float:
        """
        Search the tree from state s.
        """
        # pylint: disable=too-many-locals
        state = state.canonical()
        s = state.board.zobrist_hash()

        if state.is_terminal():
            return -self._terminal_score(state)

        if s not in self.Ps:
            # new leaf node
            # use neural net to predict current policy and value
            with torch.no_grad():
                encoded_state = self.state_encoder.encode(state)
                policy, value = self.nn(encoded_state.unsqueeze(0))
            # squeeze to remove 0th dim (batch)
            self.Ps[s] = np.exp(policy.detach().cpu().squeeze().numpy())
            v: float = value.detach().cpu().squeeze().numpy()
            # normalize policy
            self.Vs[s] = self._moves_to_vector(state.get_legal_moves())
            self._normalize_policy(s)
            if add_noise:
                self._add_noise(s)
            return -v

        # choose next state by U(s, a), recurse
        max_u, best_a = float('-inf'), -1
        for a, valid in enumerate(self.Vs[s]):
            if valid:
                u = self._compute_Usa(s, a)
                if u > max_u:
                    max_u, best_a = u, a
        move = self.game.index_to_move(best_a)
        ns = state.next(move)
        nv = self.search(ns)

        a = self.game.move_to_index(move)
        self.Qsa[(s, a)] = (self.Qsa[(s, a)] * self.Nsa[(s, a)] + nv) / (self.Nsa[(s, a)] + 1)
        self.Nsa[(s, a)] += 1
        self.Ns[s] += 1
        return -nv

    def reset(self) -> None:
        # clear all states
        self.Qsa.clear()
        self.Ps.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Vs.clear()

    def _terminal_score(self, state: GameState) -> float:
        winner = state.winner()
        if winner == self.game.canonical_player:
            return +1.
        elif winner == self.game.canonical_player.opponent:
            return -1.
        else:
            return 0.

    def _compute_Usa(self, s: int, a: int) -> float:
        c_puct = self.config['c_puct']
        n_part = np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
        return self.Qsa[(s, a)] + c_puct * self.Ps[s][a] * n_part

    def _moves_to_vector(self, moves: List[Move]) -> np.ndarray:
        """
        Converts a list of moves to a binary vector,
        where 1 denotes a move as valid and 0 as not.
        """
        move_vector = np.zeros(self.game.action_space_size)
        for m in moves:
            move_vector[self.game.move_to_index(m)] = 1
        return move_vector

    def _normalize_policy(self, s: int) -> None:
        self.Ps[s] *= self.Vs[s]
        sum_policy = np.sum(self.Ps[s])
        if sum_policy > 0:
            # re-normalize
            self.Ps[s] /= sum_policy
        else:
            # all zero, re-initialize to uniform across legal
            self.Ps[s] += self.Vs[s]
            sum_policy = np.sum(self.Ps[s])
            self.Ps[s] /= sum_policy

    def _add_noise(self, s: int, alpha: float = 0.03, epsilon: float = 0.25) -> None:
        """
        Adds Dirichlet noise to P(s, a), as described
        """
        eta = np.random.dirichlet([alpha] * self.game.action_space_size)
        self.Ps[s] = (1 - epsilon) * self.Ps[s] + epsilon * eta
