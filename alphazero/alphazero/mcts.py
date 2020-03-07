import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np
import torch

from alphazero.alphazero.nn_modules import AlphaZeroNeuralNet
from alphazero.alphazero.state_encoders.state_encoder import GameStateEncoder
from alphazero.alphazero.types import Value, State, Action
from alphazero.games.game import Game
from alphazero.games.game_state import GameState
from alphazero.games.move import Move

EPS = 1e-8
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
        self.visited_states = set()

        # Q(s, a) Q values for state-action pair
        self.Qsa: Dict[Tuple[State, Action], Value] = defaultdict(float)
        # N(s, a) visit counts for state-action pair
        self.Nsa: Dict[Tuple[State, Action], int] = defaultdict(int)
        # N(s) visit counts for state
        self.Ns: Dict[State, int] = defaultdict(int)
        # p(s) initial policies for state, returned by NN
        self.Ps: Dict[State, torch.Tensor] \
            = defaultdict(lambda: torch.zeros((self.game.action_space_size,)))
        # terminal state values
        self.Ts: Dict[State, Value] = dict()

    def get_policy(self, state: GameState, temperature: float = 1) -> List[float]:
        """
        Runs the AlphaZero MCTS search from state.
        :param state: game state to search from and return the policy for
        :param temperature: temperature for output policy vector
        :return: a probability distribution on the action space of the game
        """
        s = state.canonical()
        s_comp = s.board_zobrist_hash()
        for _ in range(self.config['num_simulations']):
            self.search(s)

        counts = [self.Nsa[(s_comp, a)] for a in range(self.game.action_space_size)]

        # convert count into a probability distribution
        counts = [c ** (1. / temperature) for c in counts]
        sum_counts = sum(counts)
        return [c / sum_counts for c in counts]

    def search(self, state: GameState) -> float:
        # pylint: disable=too-many-locals
        """
        Search the tree from state s.
        :param s: state to start the MCTS search from
        :return: the estimated value of state `s`
        """
        s = state.canonical()
        s_comp = s.board_zobrist_hash()

        if s.is_terminal():
            winner = s.winner()
            if winner == s.canonical_player:
                score = +1.
            elif winner == s.canonical_player.opponent:
                score = -1.
            else:
                score = 0.
            self.Ts[s_comp] = score

        if s_comp in self.Ts:
            return self.Ts[s_comp]

        if s_comp not in self.Ps:
            # new leaf node
            # use nn_modules to predict current policy and value
            encoded_state = self.state_encoder.encode(s)
            self.Ps[s_comp], v = self.nn(encoded_state.unsqueeze(0))
            # squeeze to remove 0th dim (batch)
            self.Ps[s_comp] = self.Ps[s_comp].squeeze()
            v = v.squeeze()
            legal_moves = s.get_legal_moves()
            legal_vector = self._moves_to_vector(legal_moves)
            self.Ps[s_comp] *= legal_vector
            sum_Ps = torch.sum(self.Ps[s_comp])
            if sum_Ps > 0:
                # re-normalize
                self.Ps[s_comp] /= sum_Ps
            else:
                self.Ps[s_comp] += legal_vector
                sum_Ps = sum(self.Ps[s_comp])
                self.Ps[s_comp] /= sum_Ps
            self.Ns[s_comp] = 0
            return -v

        max_u, best_a = float('-inf'), None
        for a in s.get_legal_moves():
            a_idx = self.game.move_to_index(a)
            u = self.Qsa[(s_comp, a_idx)] + \
                self.config['c_puct'] * self.Ps[s_comp][a_idx] \
                * np.sqrt(self.Ns[s_comp]) / (1 + self.Nsa[(s_comp, a_idx)])
            if u > max_u:
                max_u, best_a = u, a
        a = best_a
        ns = s.next(a)
        nv = self.search(ns)

        a_idx = self.game.move_to_index(a)
        self.Qsa[(s_comp, a_idx)] = \
            (self.Qsa[(s_comp, a_idx)] * self.Nsa[(s_comp, a_idx)] + nv) \
            / (self.Nsa[(s_comp, a_idx)] + 1)
        self.Nsa[(s_comp, a_idx)] += 1
        self.Ns[s_comp] += 1

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
        return move_vector.to(self.config['device'])

    def reset(self):
        # clear all states
        self.visited_states.clear()
        self.Qsa.clear()
        self.Ps.clear()
        self.Nsa.clear()
        self.Ns.clear()
        self.Ts.clear()
