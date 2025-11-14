# src/algos/espo.py
import torch
import torch.nn.functional as F

def _epanechnikov(u):
    # u = d/h; returns nonnegative kernel weights
    w = torch.clamp(1.0 - 0.5 * (u ** 2), min=0.0)  # proportional to 1 - (u^2)/2
    return w

@torch.no_grad()
def _robust_scale(v):
    # IQR / 1.349 is robust sigma; fallback to std if degenerate
    q25, q75 = torch.quantile(v, torch.tensor([0.25, 0.75], device=v.device))
    iqr = q75 - q25
    sigma = iqr / 1.349
    if not torch.isfinite(sigma) or sigma <= 1e-12:
        sigma = v.std(unbiased=False)
    return torch.clamp(sigma, min=1e-6)

def espo_step(policy, x, y0, y1, z, optimizer, memory,
              h=None, h_c=0.9, step=None,  # bandwidth or auto if None
              kl_coef=0.01, prev_logp=None,  # tiny trust region
              prior_mix=0.0, prior_beta=2.0,  # mix-in with logistic prior early
              eps=1e-6):
    """
    ESPO step with:
      • recomputed neighbor t’s from memory (no-grad),
      • proper LOO with right bank detached,
      • robust auto-bandwidth + mild annealing,
      • NW safeguard + clipping,
      • tiny KL trust-region, and an optional DPO-prior mix-in early.

    memory: dict with keys {'x','y0','y1','z','cap'} storing raw tensors on device.
    prev_logp: previous-step log-probs for KL trust region (or None to skip).
    """
    B = x.size(0)
    device = x.device

    # Current index difference t_i = log p(y1|x) - log p(y0|x)
    logp = policy(x)  # should be log-probs
    t_left = logp[torch.arange(B, device=device), y1] - logp[torch.arange(B, device=device), y0]  # [B]

    # Recompute neighbor bank under current theta (no grad)
    if memory.get('x', None) is not None and memory['x'].numel() > 0:
        with torch.no_grad():
            mb = memory
            logp_mem = policy(mb['x'])
            t_mem = (logp_mem[torch.arange(mb['x'].size(0), device=device), mb['y1']]
                     - logp_mem[torch.arange(mb['x'].size(0), device=device), mb['y0']])
            z_mem = mb['z'].float()
    else:
        t_mem = t_left.detach().new_zeros((0,))
        z_mem = z.detach().float().new_zeros((0,))

    # Right bank is detached (neighbors fixed), includes current-batch t (detached) + memory t
    t_right = torch.cat([t_left.detach(), t_mem.detach()], dim=0)  # [N = B + M]
    z_right = torch.cat([z.detach().float(), z_mem], dim=0)        # [N]

    # Auto bandwidth if needed
    if h is None:
        sigma = _robust_scale(t_right)
        n_eff = max(t_right.numel(), 1)
        # Silverman-ish start; optionally anneal by step
        h0 = float(h_c) * float(sigma) * (n_eff ** (-1.0/5.0))
        if step is not None:
            # mild decay; keep non-increasing
            h = h0 * (1.0 + 0.01 * step) ** (-0.10)
        else:
            h = h0
    h = max(float(h), 1e-6)

    # Kernel weights W_ij over right bank, with LOO on the left square
    # Distances only depend on t_left; neighbors are fixed (t_right detached)
    u = (t_left[:, None] - t_right[None, :]) / h  # [B,N]
    W = _epanechnikov(u)  # nonnegative, compact support

    # LOO: zero the diagonal in the left BxB block without in-place on a view
    if t_right.size(0) >= B:
        left = W[:, :B]
        eye = torch.eye(B, device=device, dtype=left.dtype)
        left = left * (1.0 - eye)
        W = torch.cat([left, W[:, B:]], dim=1)

    # NW safeguard normalization (fallback uniform over all non-self neighbors)
    rowsum = W.sum(dim=1, keepdim=True)  # [B,1]
    nonself = torch.ones_like(W)
    if t_right.size(0) >= B:
        nonself[:, :B] = (1.0 - eye)  # exclude self
    nonself_count = nonself.sum(dim=1, keepdim=True).clamp_min(1.0)
    fallback = nonself / nonself_count
    W = torch.where(rowsum > 1e-12, W / (rowsum + 1e-12), fallback)

    # Plug-in probability \hat g(t_i)
    g_hat = (W * z_right[None, :]).sum(dim=1).clamp(eps, 1.0 - eps)  # [B]

    # Optional mix with logistic prior early on (stabilizes when g_hat is noisy)
    if prior_mix > 0.0:
        g_prior = torch.sigmoid(prior_beta * t_left)
        g_hat = (1.0 - prior_mix) * g_prior + prior_mix * g_hat
        g_hat = g_hat.clamp(eps, 1.0 - eps)

    # Bernoulli log-likelihood
    zf = z.float()
    loss_bce = F.binary_cross_entropy(g_hat, zf)

    # Tiny TR-KL (prev -> current) to keep log-probs stable; recommended in paper
    loss_kl = 0.0
    if kl_coef and kl_coef > 0.0 and prev_logp is not None:
        with torch.no_grad():
            pi_prev = prev_logp.softmax(dim=-1)
        logpi_cur = logp  # already log-probs
        kl = (pi_prev * (pi_prev.add(1e-12).log() - logpi_cur)).sum(dim=-1).mean()
        loss_kl = kl_coef * kl

    loss = loss_bce + loss_kl

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    return {
        'loss': float(loss.item()),
        'bce': float(loss_bce.item()),
        'kl': float(loss_kl if isinstance(loss_kl, float) else loss_kl.item()),
        'h': float(h)
    }
