from collections import defaultdict
from typing import List, Dict, Tuple

import numpy as np
import torch

from alphazero.alphazero.nn import AlphaZeroNN
from alphazero.games.base import GameState, Move, Game


class MCTS:
    """
    A MCTS manages running the Monte-Carlo Tree Search algorithm
    in AlphaZero.
    """

    def __init__(self,
                 game: Game,
                 nn: AlphaZeroNN,
                 num_simulations: int,
                 c_puct: float) -> None:
        self.game = game
        self.nn = nn
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.visited_states = set()
        # TODO: Switch to compact rep for states, e.g. Zobrist hashing
        # TODO: Implement canonical boards for all three games?
        # TODO: Tutorial returns negative values. What to do in our model?
        # Q(s, a) Q values for state-action pair
        self.Qsa: Dict[Tuple[GameState, Move], float] = defaultdict(float)
        # N(s, a) visit counts for state-action pair
        self.Nsa: Dict[Tuple[GameState, Move], int] = defaultdict(int)
        # N(s) visit counts for state
        self.Ns: Dict[GameState, int] = defaultdict(int)
        # p(s) initial policies for state, returned by NN
        self.Ps: Dict[GameState, torch.Tensor] = dict()

    def get_policy(self, state: GameState, temperature: float = 1) -> List[float]:
        """
        Runs the AlphaZero MCTS search from state.
        :param state: game state to search from and return the policy for
        :param temperature: temperature for output policy vector
        :return: a probability distribution on the action space of the game
        """
        for _ in range(self.num_simulations):
            self.search(state)
        s = state
        counts = [self.Nsa[(s, self.game.index_to_move(a))]
                  for a in range(self.game.action_space_size)]

        # convert count into a probability distribution
        counts = [c ** (1. / temperature) for c in counts]
        sum_counts = np.sum(counts)
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
            # use nn to predict current policy and value
            self.Ps[s], v = self.nn(s)
            legal = self._moves_to_vector(s.get_legal_moves())
            self.Ps[s] *= legal
            sum_Ps = sum(self.Ps[s])
            if sum_Ps > 0:
                # re-normalize
                self.Ps[s] /= sum_Ps
            else:
                self.Ps[s] += legal
                sum_Ps = sum(self.Ps[s])
                self.Ps[s] /= sum_Ps
            return v

        legal = self._moves_to_vector(s.get_legal_moves())
        self.Ps[s] *= legal


        max_u, best_a = float('-inf'), None
        for a in s.get_legal_moves():
            u = self.Qsa[(s, a)] + \
                self.c_puct * self.Ps[(s, a)] * \
                np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            if u > max_u:
                max_u, best_a = u, a
        a = best_a
        ns = s.next(a)
        nv = self.search(ns)

        self.Qsa[(s, a)] = (self.Qsa[(s, a)] * self.Nsa[(s, a)] + nv) / (self.Nsa[(s, a)] + 1)
        self.Nsa[(s, a)] += 1
        self.Ns[s] += 1

        return -nv

    def _moves_to_vector(self, moves: List[Move]) -> torch.Tensor:
        """
        Converts a list of moves to a binary vector,
        where True denotes a move as valid and 0 as not.
        :param moves: the list of moves
        :return: binary vector of legal moves
        """
        move_vector = torch.zeros(self.game.action_space_size)
        for m in moves:
            move_vector[self.game.move_to_index(m)] = 1
        return move_vector
