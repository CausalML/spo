from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import torch
from torch import nn

from .utils import TrainConfig, softplus


@dataclass
class PSPOConfig:
    outer_iters: int = 8
    inner_epochs: int = 4
    iso_eps: float = 1e-4
    warm_start_epochs: int = 5


@dataclass
class OSPOConfig:
    mem_size: int = 1024
    bandwidth: float = 1.0
    update_every: int = 1
    eps: float = 1e-6
    detach_memory: bool = True
    warm_start_epochs: int = 2
    kernel: str = "gaussian"
    l2_reg: float = 0.0
    g_hat: str = "kernel"
    g_hat_lr: float = 1e-2
    g_hat_steps: int = 5


@dataclass
class RSPOConfig:
    batch_square: bool = True


class Isotonic1D:
    def __init__(self, eps: float = 1e-4):
        self.eps = eps
        self.t_knots = None
        self.y_knots = None

    def fit(self, t: np.ndarray, z: np.ndarray) -> None:
        order = np.argsort(t)
        t_sorted = t[order]
        z_sorted = z[order]

        unique_t, inverse = np.unique(t_sorted, return_inverse=True)
        counts = np.bincount(inverse)
        sums = np.bincount(inverse, weights=z_sorted)
        y = sums / counts
        w = counts.astype(np.float64)

        means = []
        weights = []
        for yi, wi in zip(y, w):
            means.append(float(yi))
            weights.append(float(wi))
            while len(means) >= 2 and means[-2] > means[-1]:
                m2 = means.pop()
                m1 = means.pop()
                w2 = weights.pop()
                w1 = weights.pop()
                new_w = w1 + w2
                new_m = (w1 * m1 + w2 * m2) / new_w
                means.append(new_m)
                weights.append(new_w)

        y_hat = np.empty_like(y, dtype=np.float64)
        idx = 0
        for mean, weight in zip(means, weights):
            block_len = int(weight)
            y_hat[idx : idx + block_len] = mean
            idx += block_len

        y_hat = np.clip(y_hat, self.eps, 1 - self.eps)
        self.t_knots = unique_t.astype(np.float32)
        self.y_knots = y_hat.astype(np.float32)

    def predict_torch(self, t: torch.Tensor) -> torch.Tensor:
        if self.t_knots is None:
            raise RuntimeError("Isotonic1D is not fit.")
        t_knots = torch.tensor(self.t_knots, device=t.device)
        y_knots = torch.tensor(self.y_knots, device=t.device)
        idx = torch.bucketize(t, t_knots)

        left = torch.clamp(idx - 1, 0, len(t_knots) - 1)
        right = torch.clamp(idx, 0, len(t_knots) - 1)
        t0 = t_knots[left]
        t1 = t_knots[right]
        y0 = y_knots[left]
        y1 = y_knots[right]
        denom = torch.clamp(t1 - t0, min=1e-6)
        w = torch.clamp((t - t0) / denom, 0.0, 1.0)
        y = y0 + w * (y1 - y0)
        return torch.clamp(y, self.eps, 1 - self.eps)


def _gather_log_probs(log_pi: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return log_pi.gather(1, y.unsqueeze(1)).squeeze(1)


def _compute_t(log_pi: torch.Tensor, log_pi_ref: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor) -> torch.Tensor:
    log_pi_y0 = _gather_log_probs(log_pi, y0)
    log_pi_y1 = _gather_log_probs(log_pi, y1)
    log_ref_y0 = log_pi_ref[y0]
    log_ref_y1 = log_pi_ref[y1]
    return (log_pi_y1 - log_ref_y1) - (log_pi_y0 - log_ref_y0)


def train_dpo(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    train_cfg: TrainConfig,
) -> nn.Module:
    model.train()
    device = train_cfg.device
    x = torch.from_numpy(data["x"]).to(device)
    y0 = torch.from_numpy(data["y0"]).to(device)
    y1 = torch.from_numpy(data["y1"]).to(device)
    z = torch.from_numpy(data["z"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    log_pi_ref = torch.full((model.k,), -torch.log(torch.tensor(float(model.k), device=device)), device=device)

    n = x.shape[0]
    for _ in range(train_cfg.epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, train_cfg.batch_size):
            idx = perm[start : start + train_cfg.batch_size]
            xb = x[idx]
            y0b = y0[idx]
            y1b = y1[idx]
            zb = z[idx]

            log_pi = model.log_probs(xb)
            t = _compute_t(log_pi, log_pi_ref, y0b, y1b)
            t_pref = torch.where(zb > 0, t, -t)
            loss = softplus(-t_pref).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def train_rspo(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    train_cfg: TrainConfig,
    rspo_cfg: RSPOConfig,
) -> nn.Module:
    model.train()
    device = train_cfg.device
    x = torch.from_numpy(data["x"]).to(device)
    y0 = torch.from_numpy(data["y0"]).to(device)
    y1 = torch.from_numpy(data["y1"]).to(device)
    z = torch.from_numpy(data["z"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    log_pi_ref = torch.full((model.k,), -torch.log(torch.tensor(float(model.k), device=device)), device=device)

    n = x.shape[0]
    for _ in range(train_cfg.epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, train_cfg.batch_size):
            idx = perm[start : start + train_cfg.batch_size]
            xb = x[idx]
            y0b = y0[idx]
            y1b = y1[idx]
            zb = z[idx]

            log_pi = model.log_probs(xb)
            t = _compute_t(log_pi, log_pi_ref, y0b, y1b)
            t_pref = torch.where(zb > 0, t, -t)
            t_matrix = t_pref.unsqueeze(1) + t_pref.unsqueeze(0)
            loss = softplus(-t_matrix).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model


def train_pspo(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    train_cfg: TrainConfig,
    pspo_cfg: PSPOConfig,
) -> nn.Module:
    model.train()
    device = train_cfg.device
    x = torch.from_numpy(data["x"]).to(device)
    y0 = torch.from_numpy(data["y0"]).to(device)
    y1 = torch.from_numpy(data["y1"]).to(device)
    z = torch.from_numpy(data["z"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    log_pi_ref = torch.full((model.k,), -torch.log(torch.tensor(float(model.k), device=device)), device=device)
    iso = Isotonic1D(eps=pspo_cfg.iso_eps)

    n = x.shape[0]
    for _ in range(pspo_cfg.warm_start_epochs):
        perm = torch.randperm(n, device=device)
        for start in range(0, n, train_cfg.batch_size):
            idx = perm[start : start + train_cfg.batch_size]
            xb = x[idx]
            y0b = y0[idx]
            y1b = y1[idx]
            zb = z[idx]

            log_pi = model.log_probs(xb)
            t = _compute_t(log_pi, log_pi_ref, y0b, y1b)
            t_pref = torch.where(zb > 0, t, -t)
            loss = softplus(-t_pref).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    for _ in range(pspo_cfg.outer_iters):
        with torch.no_grad():
            log_pi = model.log_probs(x)
            t_all = _compute_t(log_pi, log_pi_ref, y0, y1).cpu().numpy()
            iso.fit(t_all, z.cpu().numpy())

        for _ in range(pspo_cfg.inner_epochs):
            perm = torch.randperm(n, device=device)
            for start in range(0, n, train_cfg.batch_size):
                idx = perm[start : start + train_cfg.batch_size]
                xb = x[idx]
                y0b = y0[idx]
                y1b = y1[idx]
                zb = z[idx]

                log_pi = model.log_probs(xb)
                t = _compute_t(log_pi, log_pi_ref, y0b, y1b)
                p = iso.predict_torch(t)
                loss = -(zb * torch.log(p) + (1 - zb) * torch.log(1 - p)).mean()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
    return model


def _kernel_regression(
    t_query: torch.Tensor,
    t_mem: torch.Tensor,
    z_mem: torch.Tensor,
    bandwidth: float,
    std: float,
    eps: float,
    mask: torch.Tensor | None = None,
    kernel: str = "gaussian",
) -> torch.Tensor:
    scaled = (t_query.unsqueeze(1) - t_mem.unsqueeze(0)) / (bandwidth * std + eps)
    if kernel == "epanechnikov":
        weights = torch.clamp(1.0 - scaled ** 2, min=0.0)
    else:
        weights = torch.exp(-0.5 * scaled ** 2)
    if mask is not None:
        weights = weights * (1.0 - mask)
    denom = weights.sum(dim=1) + eps
    return (weights * z_mem.unsqueeze(0)).sum(dim=1) / denom




def train_ospo(
    model: nn.Module,
    data: Dict[str, np.ndarray],
    train_cfg: TrainConfig,
    ospo_cfg: OSPOConfig,
) -> nn.Module:
    model.train()
    device = train_cfg.device
    x = torch.from_numpy(data["x"]).to(device)
    y0 = torch.from_numpy(data["y0"]).to(device)
    y1 = torch.from_numpy(data["y1"]).to(device)
    z = torch.from_numpy(data["z"]).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr, weight_decay=train_cfg.weight_decay)
    log_pi_ref = torch.full((model.k,), -torch.log(torch.tensor(float(model.k), device=device)), device=device)

    n = x.shape[0]
    perm_all = torch.randperm(n, device=device)
    mem_size = n if ospo_cfg.mem_size <= 0 else min(ospo_cfg.mem_size, n)
    mem_idx = perm_all[:mem_size]
    train_idx = perm_all

    for _ in range(ospo_cfg.warm_start_epochs):
        perm = train_idx[torch.randperm(len(train_idx), device=device)]
        for start in range(0, n, train_cfg.batch_size):
            idx = perm[start : start + train_cfg.batch_size]
            xb = x[idx]
            y0b = y0[idx]
            y1b = y1[idx]
            zb = z[idx]

            log_pi = model.log_probs(xb)
            t = _compute_t(log_pi, log_pi_ref, y0b, y1b)
            t_pref = torch.where(zb > 0, t, -t)
            loss = softplus(-t_pref).mean()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    g_hat_net = None
    g_hat_opt = None
    if ospo_cfg.g_hat == "mlp":
        g_hat_net = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        ).to(device)
        g_hat_opt = torch.optim.Adam(g_hat_net.parameters(), lr=ospo_cfg.g_hat_lr)

    for epoch in range(train_cfg.epochs):
        if ospo_cfg.detach_memory and epoch % ospo_cfg.update_every == 0:
            log_pi_mem = model.log_probs(x[mem_idx])
            t_mem = _compute_t(log_pi_mem, log_pi_ref, y0[mem_idx], y1[mem_idx]).detach()
            z_mem = z[mem_idx].float()
            with torch.no_grad():
                std = float(torch.std(t_mem).clamp(min=1e-3).item())
            bw = float(ospo_cfg.bandwidth * (len(mem_idx) ** (-0.2)))
            if g_hat_net is not None:
                t_fit = t_mem.detach().unsqueeze(1)
                z_fit = z_mem.detach().unsqueeze(1)
                for _ in range(ospo_cfg.g_hat_steps):
                    pred = g_hat_net(t_fit)
                    loss_fit = torch.nn.functional.binary_cross_entropy(pred, z_fit)
                    g_hat_opt.zero_grad()
                    loss_fit.backward()
                    g_hat_opt.step()

        perm = torch.randperm(n, device=device)
        for start in range(0, n, train_cfg.batch_size):
            idx = perm[start : start + train_cfg.batch_size]
            xb = x[idx]
            y0b = y0[idx]
            y1b = y1[idx]
            zb = z[idx]

            log_pi = model.log_probs(xb)
            t = _compute_t(log_pi, log_pi_ref, y0b, y1b)

            mask = None
            if mem_size < 5000:
                mask = (idx.unsqueeze(1) == mem_idx.unsqueeze(0)).float()

            if ospo_cfg.detach_memory:
                t_mem_use = t_mem
                z_mem_use = z_mem
                bw_use = bw
                std_use = std
            else:
                log_pi_mem = model.log_probs(x[mem_idx])
                t_mem_use = _compute_t(log_pi_mem, log_pi_ref, y0[mem_idx], y1[mem_idx])
                z_mem_use = z[mem_idx].float()
                std_use = torch.std(t_mem_use).clamp(min=1e-3)
                bw_use = ospo_cfg.bandwidth * (len(mem_idx) ** (-0.2))
            if g_hat_net is not None:
                g_hat = g_hat_net(t.unsqueeze(1)).squeeze(1)
            else:
                g_hat = _kernel_regression(t, t_mem_use, z_mem_use, bw_use, std_use, ospo_cfg.eps, mask, ospo_cfg.kernel)
            g_hat = torch.clamp(g_hat, ospo_cfg.eps, 1 - ospo_cfg.eps)
            loss = -(zb * torch.log(g_hat) + (1 - zb) * torch.log(1 - g_hat)).mean()

            if ospo_cfg.l2_reg > 0:
                l2 = torch.tensor(0.0, device=device)
                for p in model.parameters():
                    l2 = l2 + torch.sum(p ** 2)
                loss = loss + ospo_cfg.l2_reg * l2

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model
