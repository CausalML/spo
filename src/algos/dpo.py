import torch
import torch.nn.functional as F

def dpo_step(policy, x, y0, y1, z, beta_train=1.0, optimizer=None):
    """
    DPO loss with uniform reference (log π_ref diff = 0).
    Loss per sample: -log σ((log π(y1)-log π(y0))/β) if z=1, and symmetric if z=0.
    """
    logp = policy(x)
    lp1 = logp[torch.arange(x.size(0)), y1]
    lp0 = logp[torch.arange(x.size(0)), y0]
    t = (lp1 - lp0) / beta_train
    # labels in {0,1} => use logistic loss on pairwise margin
    # For z=1, want t large; for z=0, want -t large
    loss = F.binary_cross_entropy_with_logits(t, z.float())
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return {'loss': loss.item()}

