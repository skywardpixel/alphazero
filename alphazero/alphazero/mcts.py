import logging
from collections import defaultdict
from typing import List, Dict, Tuple, Any

import numpy as np

from alphazero.alphazero.nn_modules import AlphaZeroNeuralNet
from alphazero.alphazero.state_encoders.state_encoder import GameStateEncoder
from alphazero.alphazero.types import Value, State, Action
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
        self.Qsa: Dict[Tuple[State, Action], Value] = defaultdict(float)
        # N(s, a) visit counts for state-action pair
        self.Nsa: Dict[Tuple[State, Action], int] = defaultdict(int)
        # N(s) visit counts for state
        self.Ns: Dict[State, int] = defaultdict(int)
        # p(s) initial policies for state, returned by NN
        self.Ps: Dict[State, np.ndarray] = dict()

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
            self.search(state)

        counts = np.array([self.Nsa[(s, a)]
                           for a in range(self.game.action_space_size)])

        # convert count into a probability distribution
        if temperature == 0:
            # return one-hot vector for policy
            policy = np.eye(self.game.action_space_size)[np.argmax(counts)]
            return policy
        counts = counts ** (1. / temperature)
        return counts / np.sum(counts)

    def search(self, state: GameState) -> float:
        # pylint: disable=too-many-locals
        """
        Search the tree from state s.
        :param state: state to start the MCTS search from
        :return: the estimated *negative* value of `state`
        """
        state = state.canonical()
        s = state.board.zobrist_hash()

        if state.is_terminal():
            winner = state.winner()
            if winner == self.game.canonical_player:
                score = +1.
            elif winner == self.game.canonical_player.opponent:
                score = -1.
            else:
                score = 0.
            return -score

        if s not in self.Ps:
            # new leaf node
            # use neural net to predict current policy and value
            encoded_state = self.state_encoder.encode(state)
            policy, value = self.nn(encoded_state.unsqueeze(0))

            # squeeze to remove 0th dim (batch)
            self.Ps[s] = np.exp(policy.detach().cpu().squeeze().numpy())
            v = value.detach().cpu().squeeze().numpy()

            # normalize policy
            legal_moves = state.get_legal_moves()
            legal_vector = self._moves_to_vector(legal_moves)
            self.Ps[s] *= legal_vector
            sum_Ps = np.sum(self.Ps[s])
            if sum_Ps > 0:
                # re-normalize
                self.Ps[s] /= sum_Ps
            else:
                # all zero, re-initialize to uniform across legal
                self.Ps[s] += legal_vector
                sum_Ps = np.sum(self.Ps[s])
                self.Ps[s] /= sum_Ps

            return -v

        # choose next state by U(s, a), recurse
        c_puct = self.config['c_puct']
        max_u, best_move = float('-inf'), None
        for move in state.get_legal_moves():
            a = self.game.move_to_index(move)
            u = self.Qsa[(s, a)] + \
                c_puct * self.Ps[s][a] * np.sqrt(self.Ns[s]) / (1 + self.Nsa[(s, a)])
            if u > max_u:
                max_u, best_move = u, move
        move = best_move
        ns = state.next(move)
        nv = self.search(ns)

        a = self.game.move_to_index(move)
        self.Qsa[(s, a)] = (self.Qsa[(s, a)] * self.Nsa[(s, a)] + nv) / (self.Nsa[(s, a)] + 1)
        self.Nsa[(s, a)] += 1
        self.Ns[s] += 1

        return -nv

    def _moves_to_vector(self, moves: List[Move]) -> np.ndarray:
        """
        Converts a list of moves to a binary vector,
        where 1 denotes a move as valid and 0 as not.
        :param moves: the list of moves
        :return: binary vector of legal moves
        """
        move_vector = np.zeros(self.game.action_space_size)
        for m in moves:
            move_vector[self.game.move_to_index(m)] = 1
        return move_vector

    def reset(self):
        # clear all states
        self.Qsa.clear()
        self.Ps.clear()
        self.Nsa.clear()
        self.Ns.clear()
