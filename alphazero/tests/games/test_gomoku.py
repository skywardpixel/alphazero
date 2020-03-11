import unittest

import numpy as np
import torch

from alphazero.games.gomoku import GomokuGame, GomokuMove


class GomokuTestCase(unittest.TestCase):
    def test_canonical(self):
        game = GomokuGame(9)
        self.assertEqual(str(game.state.canonical().board), str(game.state.board))
        game.play(GomokuMove(0, 0))
        self.assertNotEqual(str(game.state.canonical().board), str(game.state.board))
        game.play(GomokuMove(2, 2))
        self.assertEqual(str(game.state.canonical().board), str(game.state.board))

    def test_symmetry(self):
        game = GomokuGame(3)
        # pylint: disable=not-callable
        state = torch.tensor([
            [
                [1, 2, 3],
                [4, 5, 6],
                [7, 8, 9],
            ],
            [
                [10, 20, 30],
                [40, 50, 60],
                [70, 80, 90],
            ],
            [
                [100, 200, 300],
                [400, 500, 600],
                [700, 800, 900],
            ],
        ])
        policy = np.array([
            11, 22, 33,
            44, 55, 66,
            77, 88, 99,
        ])
        symmetries = game.symmetries(state, policy)
        for s, p in symmetries:
            self.assertTrue(np.array_equal(s[0].flatten().numpy() * 11, p))
            self.assertTrue(torch.equal(s[0] * 10, s[1]))
            self.assertTrue(torch.equal(s[1] * 10, s[2]))


if __name__ == '__main__':
    unittest.main()
