import torch
import math

def avg_kl_to_uniform(logits, beta, reduce='mean'):
    """
    Given uncalibrated index h := f'(πθ/πref) for KL, the calibrated policy is:
      πβ(y|x) ∝ π_ref(y|x) * exp( (1/β) * h(y|x) ).
    For uniform π_ref, this reduces to softmax((1/β) * h).
    Here, for our trainers, we use h = log πθ(y|x) - log π_ref(y|x),
    but with uniform π_ref, h = log πθ + const; constants cancel in softmax.
    We therefore use logits as h.
    """
    # calibrated policy
    p = torch.softmax(logits / beta, dim=-1)  # πβ
    # KL(πβ || U) = sum p log(p * |Y|)
    n_actions = p.size(-1)
    kl = (p * (torch.log(p + 1e-12) + math.log(n_actions))).sum(dim=-1)
    return kl.mean() if reduce == 'mean' else kl

def calibrate_beta_for_kappa(logits_fn, x_calib, kappa, beta_min=0.01, beta_max=100.0, tol=1e-4, max_steps=50):
    """
    Bisection on β to solve E[KL(πβ(x)||U)] = κ.
    logits_fn: callable x -> uncalibrated logits h(x) (e.g., log-prob-based score)
    """
    with torch.no_grad():
        h = logits_fn(x_calib)  # shape [m, n_actions]
        lo, hi = beta_min, beta_max
        for _ in range(max_steps):
            mid = (lo + hi) / 2.0
            val = avg_kl_to_uniform(h, mid).item()
            if abs(val - kappa) < tol:
                return mid
            if val > kappa:
                # too peaky -> increase β
                lo = mid
            else:
                # too flat -> decrease β
                hi = mid
        return (lo + hi) / 2.0

