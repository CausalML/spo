import torch
import torch.nn.functional as F

def sft_step(policy, x, y_w, optimizer):
    # Supervise on winner only
    logp = policy.logprobs(x)  # [B,A]
    loss = F.nll_loss(logp, y_w)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return {'loss': loss.item()}

