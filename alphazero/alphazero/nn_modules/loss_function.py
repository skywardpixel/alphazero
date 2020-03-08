import torch
from torch.nn.modules.loss import _Loss


class AlphaZeroLoss(_Loss):
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
                             torch.log(p).view(batch_size, action_space_size, 1))
        pi_loss = pi_loss.flatten()
        return torch.mean(v_loss + pi_loss)
