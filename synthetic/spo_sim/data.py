from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch

from .models import MLPPolicy


@dataclass
class DGPConfig:
    d: int = 15
    k: int = 8
    nu: float = 2.0
    s: float = 0.0
    tau: float = 1.0


def link_function(u: torch.Tensor, s: float, tau: float) -> torch.Tensor:
    return 0.5 * torch.sigmoid((u - s) / tau) + 0.5 * torch.sigmoid((u + s) / tau)


def generate_teacher(seed: int, d: int, k: int, hidden_sizes: Tuple[int, ...]) -> MLPPolicy:
    torch.manual_seed(seed)
    model = MLPPolicy(d, k, hidden_sizes)
    model.eval()
    return model


def sample_dataset(
    seed: int,
    n: int,
    m: int,
    dgp: DGPConfig,
    teacher: MLPPolicy,
    device: str = "cpu",
):
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(n, dgp.d)).astype(np.float32)
    x_eval = rng.normal(size=(m, dgp.d)).astype(np.float32)

    y0 = rng.integers(low=0, high=dgp.k, size=n, dtype=np.int64)
    y1 = rng.integers(low=0, high=dgp.k, size=n, dtype=np.int64)

    with torch.no_grad():
        tx = torch.from_numpy(x_train).to(device)
        log_pi = teacher.log_probs(tx)
        log_pi_ref = -torch.log(torch.tensor(float(dgp.k), device=device))
        r_y0 = dgp.nu * (log_pi.gather(1, torch.from_numpy(y0).to(device).unsqueeze(1)).squeeze(1) - log_pi_ref)
        r_y1 = dgp.nu * (log_pi.gather(1, torch.from_numpy(y1).to(device).unsqueeze(1)).squeeze(1) - log_pi_ref)
        diff = r_y1 - r_y0
        p = link_function(diff, dgp.s, dgp.tau)
        z = torch.bernoulli(p).cpu().numpy().astype(np.int64)

    data = {
        "x": x_train,
        "y0": y0,
        "y1": y1,
        "z": z,
        "x_eval": x_eval,
    }
    return data
