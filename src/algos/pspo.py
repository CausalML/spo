import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict

class MonoLink(nn.Module):
    """
    Differentiable monotone 1D link via a positive mixture of increasing bases:
      logit g(t) = a0 + sum_k softplus(w_k) * softplus((t - c_k)/gamma)
    We return *logits* via forward_logit(); use BCE-with-logits for stability.
    """
    def __init__(self, n_knots=16, t_min=-3.0, t_max=3.0, gamma=0.5, train_knots: bool=False):
        super().__init__()
        c = torch.linspace(t_min, t_max, n_knots)
        if train_knots:
            self.c = nn.Parameter(c)
        else:
            self.register_buffer("c", c)
        self.w = nn.Parameter(torch.zeros(n_knots))   # unconstrained; softplus -> positive slopes
        self.a0 = nn.Parameter(torch.tensor(0.0))
        self.gamma = float(gamma)

    def _bases(self, t: torch.Tensor) -> torch.Tensor:
        # strictly increasing in t
        return F.softplus((t[:, None] - self.c[None, :]) / self.gamma)

    def forward_logit(self, t: torch.Tensor) -> torch.Tensor:
        s = self._bases(t)                              # [B,K]
        w_pos = F.softplus(self.w)[None, :]             # [1,K] >= 0
        return self.a0 + (w_pos * s).sum(dim=-1)        # [B]

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.forward_logit(t))


@torch.no_grad()
def _append_link_memory(buf: Dict[str, torch.Tensor], t: torch.Tensor, z: torch.Tensor, cap: int):
    if buf["t"] is None:
        buf["t"] = t.detach()
        buf["z"] = z.detach().float()
    else:
        buf["t"] = torch.cat([buf["t"], t.detach()], dim=0)[-cap:]
        buf["z"] = torch.cat([buf["z"], z.detach().float()], dim=0)[-cap:]


def pspo_step(
    policy: nn.Module,
    link_model: MonoLink,
    tnorm,                                   # RobustTNormalizer instance
    x: torch.Tensor, y0: torch.Tensor, y1: torch.Tensor, z: torch.Tensor,
    optimizer_theta: torch.optim.Optimizer,
    optimizer_link: torch.optim.Optimizer,
    alt_steps: int = 2,
    link_memory: Optional[Dict[str, torch.Tensor]] = None,
    link_mem_cap: int = 50_000,
    link_l2: float = 1e-4,                   # tiny penalty on positive slopes
    kl_coef: float = 1e-2,                   # small TR-KL(prev||cur)
    prev_logp: Optional[torch.Tensor] = None,
    clip_grad_norm: Optional[float] = 1.0,
    max_link_points: int = 8192,
) -> dict:
    """
    PSPO alternating step with robust IQR normalization of t (Option A).
      (A) Fit monotone link on normalized (t_detach, z) using BCE-with-logits (batch + memory).
      (B) Update policy θ with link frozen on normalized t(θ) + tiny KL(prev||cur).
    """
    # Current log-probs and index difference
    logp = policy(x)  # should return log-probs
    t = logp[torch.arange(x.size(0), device=x.device), y1] - logp[torch.arange(x.size(0), device=x.device), y0]

    # Build/update normalization stats from pool (batch ⊕ memory)
    t_batch = t.detach()
    if link_memory is not None and link_memory.get("t", None) is not None:
        t_pool = torch.cat([link_memory["t"], t_batch], dim=0)
    else:
        t_pool = t_batch
    tnorm.update_from(t_pool)  # no grad

    # Normalized t for link fit (detach) and for policy (with grad)
    t_batch_n = tnorm.normalize(t_batch)
    t_n       = tnorm.normalize(t)

    # Compose link training set: current batch (normalized) + memory (normalized)
    if link_memory is not None and link_memory.get("t", None) is not None:
        t_mem_n = tnorm.normalize(link_memory["t"])
        t_train = torch.cat([t_batch_n, t_mem_n], dim=0)
        z_train = torch.cat([z.detach().float(), link_memory["z"]], dim=0)
    else:
        t_train, z_train = t_batch_n, z.detach().float()

    # Optionally subsample the link training set
    if t_train.numel() > max_link_points:
        idx = torch.randint(0, t_train.numel(), (max_link_points,), device=t_train.device)
        t_train = t_train[idx]
        z_train = z_train[idx]

    # ---- (A) Update link (few inner steps) ----
    loss_link = None
    for _ in range(max(1, int(alt_steps))):
        v = link_model.forward_logit(t_train)                 # logits
        loss_link = F.binary_cross_entropy_with_logits(v, z_train)
        loss_link = loss_link + link_l2 * F.softplus(link_model.w).sum()  # mild slope control
        optimizer_link.zero_grad(set_to_none=True)
        loss_link.backward()
        optimizer_link.step()

    # ---- (B) Update policy θ with link frozen ----
    v_theta = link_model.forward_logit(t_n)                   # logits; grads flow through t(θ)
    loss_theta = F.binary_cross_entropy_with_logits(v_theta, z.float())

    # tiny TR-KL to keep policy stable
    if kl_coef and prev_logp is not None:
        with torch.no_grad():
            pi_prev = prev_logp.softmax(dim=-1)
        kl = torch.sum(pi_prev * (pi_prev.add(1e-12).log() - logp), dim=-1).mean()
        loss_theta = loss_theta + kl_coef * kl

    optimizer_theta.zero_grad(set_to_none=True)
    loss_theta.backward()
    if clip_grad_norm is not None:
        torch.nn.utils.clip_grad_norm_(policy.parameters(), max_norm=clip_grad_norm)
    optimizer_theta.step()

    # Update link memory with normalized batch t’s (we store *raw* t is also OK; here we store normalized)
    if link_memory is not None:
        _append_link_memory(link_memory, t_batch_n, z.detach().float(), link_mem_cap)

    return {
        "loss_theta": float(loss_theta.detach().item()),
        "loss_link": float(loss_link.detach().item()) if loss_link is not None else None,
    }
