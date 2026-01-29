from __future__ import annotations

import torch


def compute_kl(logprob_beta: torch.Tensor, logprob_ref: torch.Tensor) -> float:
    diff = logprob_beta - logprob_ref
    diff = torch.nan_to_num(diff, nan=0.0, posinf=0.0, neginf=0.0)
    return diff.mean().item()


def summarize_rewards(rewards: torch.Tensor) -> float:
    return rewards.mean().item()
