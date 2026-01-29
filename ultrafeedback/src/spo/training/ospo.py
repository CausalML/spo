from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import torch

from .utils import batch_logprobs


@dataclass
class OSPOMemory:
    t_values: torch.Tensor
    z_values: torch.Tensor


def compute_t(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer,
    prompts: List[str],
    responses_0: List[str],
    responses_1: List[str],
    max_prompt_length: int,
    max_response_length: int,
    device: torch.device,
) -> torch.Tensor:
    logp_y0 = batch_logprobs(model, tokenizer, prompts, responses_0, max_prompt_length, max_response_length, device)
    logp_y1 = batch_logprobs(model, tokenizer, prompts, responses_1, max_prompt_length, max_response_length, device)
    with torch.no_grad():
        logp_ref_y0 = batch_logprobs(ref_model, tokenizer, prompts, responses_0, max_prompt_length, max_response_length, device)
        logp_ref_y1 = batch_logprobs(ref_model, tokenizer, prompts, responses_1, max_prompt_length, max_response_length, device)
    return (logp_y1 - logp_ref_y1) - (logp_y0 - logp_ref_y0)


def init_memory() -> OSPOMemory:
    return OSPOMemory(t_values=torch.empty(0), z_values=torch.empty(0))


def update_memory(memory: OSPOMemory, t_batch: torch.Tensor, z_batch: torch.Tensor, max_size: int) -> OSPOMemory:
    t_new = torch.cat([memory.t_values, t_batch.detach().cpu()])
    z_new = torch.cat([memory.z_values, z_batch.detach().cpu()])
    if len(t_new) > max_size:
        t_new = t_new[-max_size:]
        z_new = z_new[-max_size:]
    return OSPOMemory(t_values=t_new, z_values=z_new)


def kernel_regression(
    t_query: torch.Tensor,
    memory: OSPOMemory,
    kernel: str,
    bandwidth: float,
) -> torch.Tensor:
    if len(memory.t_values) == 0:
        return torch.full_like(t_query, 0.5) + 0.0 * t_query
    t_mem = memory.t_values.to(t_query.device).detach()
    z_mem = memory.z_values.to(t_query.device).detach()
    sigma = torch.std(t_mem, unbiased=False) + 1e-6
    u = (t_query.unsqueeze(1) - t_mem.unsqueeze(0)) / (bandwidth * sigma)
    if kernel == "gaussian":
        weights = torch.exp(-0.5 * u * u)
    elif kernel == "epanechnikov":
        mask = torch.abs(u) <= 1
        weights = torch.zeros_like(u)
        weights[mask] = 0.75 * (1 - u[mask] ** 2)
    else:
        raise ValueError(f"Unknown kernel: {kernel}")
    denom = weights.sum(dim=1) + 1e-6
    numer = (weights * z_mem.unsqueeze(0)).sum(dim=1)
    g_hat = numer / denom
    return torch.clamp(g_hat, 1e-4, 1 - 1e-4)


def ospo_loss(
    t_values: torch.Tensor,
    z: torch.Tensor,
    g_hat: torch.Tensor,
) -> torch.Tensor:
    z = z.float()
    return -(z * torch.log(g_hat) + (1 - z) * torch.log(1 - g_hat)).mean()
