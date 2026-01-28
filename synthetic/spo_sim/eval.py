from typing import List, Tuple

import numpy as np
import torch

from .data import DGPConfig
from .models import MLPPolicy


def evaluate_policy(
    model: MLPPolicy,
    teacher: MLPPolicy,
    x_eval: np.ndarray,
    dgp: DGPConfig,
    beta_grid: List[float],
    zeta: float = 1.0,
    device: str = "cpu",
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    teacher.eval()

    x_t = torch.from_numpy(x_eval).to(device)
    log_pi_ref = -torch.log(torch.tensor(float(dgp.k), device=device))
    log_pi_ref_vec = torch.full((dgp.k,), log_pi_ref, device=device)

    with torch.no_grad():
        log_pi_theta = model.log_probs(x_t)
        log_pi_theta = log_pi_ref_vec + zeta * (log_pi_theta - log_pi_ref_vec)
        log_pi_star = teacher.log_probs(x_t)
        r_star = dgp.nu * (log_pi_star - log_pi_ref)

    rewards = []
    kls = []
    for beta in beta_grid:
        invb = 1.0 / beta
        with torch.no_grad():
            log_pi_beta = (1 - invb) * log_pi_ref_vec + invb * log_pi_theta
            log_pi_beta = log_pi_beta - torch.logsumexp(log_pi_beta, dim=1, keepdim=True)
            pi_beta = torch.exp(log_pi_beta)
            avg_reward = (pi_beta * r_star).sum(dim=1).mean().item()
            avg_kl = (pi_beta * (log_pi_beta - log_pi_ref)).sum(dim=1).mean().item()
        rewards.append(avg_reward)
        kls.append(avg_kl)

    return np.array(rewards), np.array(kls)
