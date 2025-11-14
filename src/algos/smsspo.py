import torch
import torch.nn.functional as F

def smsspo_step(policy, x, y0, y1, z, optimizer, sigma=0.5):
    """
    Smoothed max-score objective: sum (2z-1) * K(t / σ).
    """
    logp = policy(x)
    t = logp[torch.arange(x.size(0)), y1] - logp[torch.arange(x.size(0)), y0]
    s = 2 * z.float() - 1.0
    K = torch.sigmoid(t / sigma)
    loss = - (s * K).mean()  # ascend, so minimize negative
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {'loss': loss.item()}

