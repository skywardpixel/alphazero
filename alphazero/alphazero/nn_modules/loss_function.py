import torch
from torch import nn


class AlphaZeroLoss(nn.Module):
    """
    Computes the loss for the AlphaZero neural network.
    """

    def forward(self, input, target):
        # pylint: disable=arguments-differ, redefined-builtin
        (p, v), (pi, z) = input, target
        v = v.flatten()
        v_loss = (v - z) ** 2
        batch_size, action_space_size = pi.shape
        pi_loss = -torch.bmm(pi.view(batch_size, 1, action_space_size),
                             p.view(batch_size, action_space_size, 1))
        pi_loss = pi_loss.flatten()
        return torch.mean(v_loss + pi_loss)
