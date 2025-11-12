import torch
import torch.nn as nn
import torch.nn.functional as F

class MonoLink(nn.Module):
    """
    Differentiable monotone 1D link g(t) in (0,1):
      g(t) = sigmoid( a0 + sum_k softplus(w_k) * softplus((t - c_k)/gamma) )
    """
    def __init__(self, n_knots=16, t_min=-6., t_max=6., gamma=0.5):
        super().__init__()
        self.c = nn.Parameter(torch.linspace(t_min, t_max, n_knots))
        self.w = nn.Parameter(torch.zeros(n_knots))
        self.a0 = nn.Parameter(torch.tensor(0.0))
        self.gamma = gamma

    def forward(self, t):
        # [B] -> [B]
        s = torch.nn.functional.softplus((t[:, None] - self.c[None, :]) / self.gamma)  # [B,K]
        inc = torch.nn.functional.softplus(self.w)[None, :] * s  # positive increments
        val = self.a0 + inc.sum(dim=-1)
        return torch.sigmoid(val).clamp(1e-6, 1 - 1e-6)

def pspo_step(policy, link_model, x, y0, y1, z, optimizer_theta, optimizer_link, alt_steps=1):
    """
    Alternating profile-like step:
      1) Fit monotone g to (t, z) by maximizing log-likelihood (few steps).
      2) Update θ with g frozen, maximizing sum log g(t(θ)).
    """
    # index difference under KL/uniform: t = log π(y1) - log π(y0)
    logp = policy(x)
    t = logp[torch.arange(x.size(0)), y1] - logp[torch.arange(x.size(0)), y0]

    # Step A: update link g (few inner steps)
    for _ in range(alt_steps):
        g = link_model(t.detach())  # detach t for profile step
        loss_link = - (z * torch.log(g) + (1 - z) * torch.log(1 - g)).mean()
        optimizer_link.zero_grad(set_to_none=True)
        loss_link.backward()
        optimizer_link.step()

    # Step B: update θ (envelope-like; keep g fixed but allow gradient through t)
    g = link_model(t)    # now flows gradient through t(θ)
    loss_theta = - (z * torch.log(g) + (1 - z) * torch.log(1 - g)).mean()
    optimizer_theta.zero_grad(set_to_none=True)
    loss_theta.backward()
    optimizer_theta.step()

    return {'loss_theta': loss_theta.item(), 'loss_link': loss_link.item()}

