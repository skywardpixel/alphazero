from typing import Dict

import torch
from torch.nn.modules.loss import _Loss


class AlphaZeroLoss(_Loss):
    """
    Computes the loss for the AlphaZero neural network.
    """
    def forward(self, inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        # pylint: disable=arguments-differ
        vs, zs, pis, ps = inputs['vs'], inputs['zs'], inputs['pis'], inputs['ps']
        v_loss = (vs - zs) ** 2
        pi_loss = -torch.dot(pis, torch.log(ps))
        return v_loss + pi_loss
