from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from .utils import batch_logprobs
from ..utils.isotonic import isotonic_predict, pava


@dataclass
class PSPOState:
    x_hat: np.ndarray
    y_hat: np.ndarray


def fit_isotonic(t_values: np.ndarray, z_values: np.ndarray) -> PSPOState:
    x_hat, y_hat = pava(t_values, z_values)
    y_hat = np.clip(y_hat, 1e-4, 1 - 1e-4)
    return PSPOState(x_hat=x_hat, y_hat=y_hat)


def pspo_loss(
    t_values: torch.Tensor,
    z: torch.Tensor,
    state: PSPOState,
) -> torch.Tensor:
    x_hat = torch.tensor(state.x_hat, device=t_values.device, dtype=t_values.dtype)
    y_hat = torch.tensor(state.y_hat, device=t_values.device, dtype=t_values.dtype)
    # Smooth interpolation for differentiability
    soft_tau = 0.1
    distances = torch.abs(t_values.unsqueeze(1) - x_hat.unsqueeze(0))
    weights = torch.softmax(-distances / soft_tau, dim=1)
    psi_t = torch.sum(weights * y_hat.unsqueeze(0), dim=1)
    # Blend with a small logistic term to avoid flat gradients when isotonic fit is constant
    mix = 0.05
    psi_t = (1 - mix) * psi_t + mix * torch.sigmoid(t_values)
    psi_t = torch.clamp(psi_t, 1e-6, 1 - 1e-6)
    z = z.float()
    return -(z * torch.log(psi_t) + (1 - z) * torch.log(1 - psi_t)).mean()


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


def update_isotonic_state(
    model: torch.nn.Module,
    ref_model: torch.nn.Module,
    tokenizer,
    dataloader,
    max_prompt_length: int,
    max_response_length: int,
    device: torch.device,
) -> PSPOState:
    t_values = []
    z_values = []
    model.eval()
    with torch.no_grad():
        for batch in dataloader:
            t = compute_t(
                model,
                ref_model,
                tokenizer,
                batch["prompt"],
                batch["response_0"],
                batch["response_1"],
                max_prompt_length,
                max_response_length,
                device,
            )
            t_values.append(t.detach().float().cpu().numpy())
            z_values.append(np.array(batch["z"]))
    t_concat = np.concatenate(t_values)
    z_concat = np.concatenate(z_values).astype(np.float32)
    return fit_isotonic(t_concat, z_concat)
