from __future__ import annotations

import torch


def rspo_loss(s_values: torch.Tensor) -> torch.Tensor:
    # s_values: (B,) preference score for each example
    # Use all pair combinations inside batch
    s_i = s_values.unsqueeze(0)
    s_j = s_values.unsqueeze(1)
    pair_sum = s_i + s_j
    loss = torch.logaddexp(torch.zeros_like(pair_sum), -pair_sum)
    return loss.mean()
