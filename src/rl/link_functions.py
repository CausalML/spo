import torch
import torch.nn.functional as F

def logistic_cdf(u):
    return torch.sigmoid(u)

def probit_cdf(u):
    # approximate Φ with erf
    return 0.5 * (1.0 + torch.erf(u / (2.0**0.5)))

def beta_sigmoid_cdf(u, a=2.0, b=5.0):
    """
    Non-standard link: Psi(u) = BetaCDF(sigmoid(u); a, b).
    Monotone, flexible, not logistic/probit.
    """
    s = torch.sigmoid(u).clamp(1e-6, 1-1e-6)
    # Beta CDF via incomplete beta using regularized incomplete beta approximation
    # Use torch.special.betainc if available
    try:
        from torch.special import betainc
        # Regularized incomplete beta I_s(a,b)
        return betainc(torch.tensor(a, device=u.device), torch.tensor(b, device=u.device), s)
    except Exception:
        # Fallback: smooth polynomial approx (adequate for simulation)
        # Use logit-normal CDF power transform as fallback: s**a / (s**a + (1-s)**b)
        return (s.pow(a)) / (s.pow(a) + (1 - s).pow(b) + 1e-12)

def double_sigmoid_cdf(u, a=-1.25, b=1.25):
    """
    Non-standard link: Psi(u) = sigmoid(u+a)/2 + sigmoid(u-b)/2.
    Monotone, flexible, not logistic/probit.
    """
    return 0.5*torch.sigmoid(4*(u+a)).clamp(1e-8, 1-1e-8)+0.5*torch.sigmoid(4*(u+b)).clamp(1e-8, 1-1e-8)
    