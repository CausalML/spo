import torch
import torch.nn.functional as F

def gaussian_kernel(d, h):
    return torch.exp(-0.5 * (d / (h + 1e-12))**2)

def espo_step(policy, x, y0, y1, z, optimizer, h=0.5, memory=None):
    """
    ESPO: plug-in estimate g(t) = P(z=1 | t) via kernel smoothing on t.
    We compute a leave-one-out approximation within a minibatch + optional memory bank.
    """
    logp = policy.logprobs(x)
    t = logp[torch.arange(x.size(0)), y1] - logp[torch.arange(x.size(0)), y0]  # [B]
    t_all = t.detach()
    z_all = z.detach().float()

    if memory is not None and memory['t'] is not None:
        t_all = torch.cat([t_all, memory['t']], dim=0)
        z_all = torch.cat([z_all, memory['z']], dim=0)

    # Kernel weights: [B, N]
    with torch.no_grad():
        dists = t[:, None] - t_all[None, :]
        W = gaussian_kernel(dists, h)
        W = W / (W.sum(dim=1, keepdim=True) + 1e-12)
        g_hat = (W * z_all[None, :]).sum(dim=1).clamp(1e-4, 1 - 1e-4)

    # Now treat g_hat(t) as target prob and optimize θ by Bernoulli log-likelihood
    loss = - (z.float() * torch.log(g_hat) + (1 - z.float()) * torch.log(1 - g_hat)).mean()
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

    # update memory bank
    if memory is not None:
        tb = t.detach()
        zb = z.detach().float()
        if memory['t'] is None:
            memory['t'] = tb
            memory['z'] = zb
        else:
            memory['t'] = torch.cat([memory['t'], tb], dim=0)[-memory['cap']:]
            memory['z'] = torch.cat([memory['z'], zb], dim=0)[-memory['cap']:]

    return {'loss': loss.item()}

