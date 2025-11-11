import torch
import math

def expected_true_reward(policy_probs, x, W):
    """
    E_y r*(x,y) under learned policy.
    r*(x,y) = <W_y, x>
    policy_probs: [B, A]
    x: [B, D]
    W: [A, D]
    """
    by = torch.einsum('ad,bd->ba', W, x)  # [B,A] rewards
    return (policy_probs * by).sum(dim=-1).mean().item()

def kl_to_uniform(policy_probs):
    n_actions = policy_probs.size(-1)
    kl = (policy_probs * (torch.log(policy_probs + 1e-12) + math.log(n_actions))).sum(dim=-1)
    return kl.mean().item()

