from __future__ import annotations

from typing import Dict, List

import torch
import torch.nn.functional as F


def dpo_loss(
    logp_y1: torch.Tensor,
    logp_y0: torch.Tensor,
    logp_ref_y1: torch.Tensor,
    logp_ref_y0: torch.Tensor,
) -> torch.Tensor:
    log_ratio_pref = logp_y1 - logp_ref_y1
    log_ratio_other = logp_y0 - logp_ref_y0
    loss = torch.logaddexp(torch.zeros_like(log_ratio_pref), log_ratio_other - log_ratio_pref)
    return loss.mean()
