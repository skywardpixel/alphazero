from collections import defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
import torch

from alphazero.alphazero.nn_modules import AlphaZeroNeuralNet
from alphazero.alphazero.state_encoders.state_encoder import GameStateEncoder
from alphazero.games.game import Game
from alphazero.games.game_state import GameState
from alphazero.games.move import Move

EPS = 1e-8


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
        self.visited_states = set()
        # TODO: Switch to compact rep for states, e.g. Zobrist hashing
        # TODO: Implement canonical boards for all three games?
        # TODO: Tutorial returns negative values. What to do in our model?
        # Q(s, a) Q values for state-action pair
        self.Qsa: Dict[Tuple[GameState, int], float] = defaultdict(float)
        # N(s, a) visit counts for state-action pair
        self.Nsa: Dict[Tuple[GameState, int], int] = defaultdict(int)
        # N(s) visit counts for state
        self.Ns: Dict[GameState, int] = defaultdict(int)
        # p(s) initial policies for state, returned by NN
        self.Ps: Dict[GameState, torch.Tensor] \
            = defaultdict(lambda: torch.zeros((self.game.action_space_size,)))

    def get_policy(self, state: GameState, temperature: float = 1) -> List[float]:
        """
        Runs the AlphaZero MCTS search from state.
        :param state: game state to search from and return the policy for
        :param temperature: temperature for output policy vector
        :return: a probability distribution on the action space of the game
        """
        for _ in range(self.config['num_simulations']):
            self.search(state)
        s = state
        counts = [self.Nsa[(s, a)] for a in range(self.game.action_space_size)]

        # convert count into a probability distribution
        counts = [c ** (1. / temperature) for c in counts]
        sum_counts = sum(counts)
        return [c / sum_counts for c in counts]

    def search(self, state: GameState) -> float:
        """
        Search the tree from state s.
        :param s: state to start the MCTS search from
        :return: the estimated value of state `s`
        """
        s = state
        # TODO: is s already visited?

        if s not in self.Ps:
            # new leaf node
            # use nn_modules to predict current policy and value
            encoded_state = self.state_encoder.encode(s)
            self.Ps[s], v = self.nn(encoded_state)
            # squeeze to remove 0th dim (batch)
            self.Ps[s] = self.Ps[s].squeeze()
            v = v.squeeze()
            legal_moves = s.get_legal_moves()
            legal_vector = self._moves_to_vector(legal_moves)
            self.Ps[s] *= legal_vector
            sum_Ps = torch.sum(self.Ps[s])
            if sum_Ps > 0:
                # re-normalize
                self.Ps[s] /= sum_Ps
            else:
                self.Ps[s] += legal_vector
                sum_Ps = sum(self.Ps[s])
                self.Ps[s] /= sum_Ps
            self.Ns[s] = 0
            return -v

        max_u, best_a = float('-inf'), None
        for a in s.get_legal_moves():
            a_idx = self.game.move_to_index(a)
            u = self.Qsa[(s, a_idx)] + \
                self.config['c_puct'] * self.Ps[s][a_idx] \
                * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a_idx)])
            if u > max_u:
                max_u, best_a = u, a
        a = best_a
        ns = s.next(a)
        nv = self.search(ns)

        a_idx = self.game.move_to_index(a)
        self.Qsa[(s, a_idx)] = (self.Qsa[(s, a_idx)] * self.Nsa[(s, a_idx)] + nv) \
                               / (self.Nsa[(s, a_idx)] + 1)
        self.Nsa[(s, a_idx)] += 1
        self.Ns[s] += 1

        return -nv

    def _moves_to_vector(self, moves: List[Move]) -> torch.Tensor:
        """
        Converts a list of moves to a binary vector,
        where 1 denotes a move as valid and 0 as not.
        :param moves: the list of moves
        :return: binary vector of legal moves
        """
        move_vector = torch.zeros(self.game.action_space_size)
        for m in moves:
            move_vector[self.game.move_to_index(m)] = 1
        return move_vector

    def reset(self):
        # TODO: clear all states
        self.visited_states.clear()
        self.Qsa.clear()
        self.Ps.clear()
        self.Nsa.clear()
        self.Ns.clear()
