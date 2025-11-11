import torch
import numpy as np
from .utils import set_seed
from src.rl.link_functions import beta_sigmoid_cdf

def make_reward_weights(n_actions, feat_dim, device):
    # Fixed true reward direction per action
    W = torch.randn(n_actions, feat_dim, device=device)
    W = W / (W.norm(dim=1, keepdim=True) + 1e-8)
    return W

def true_reward(x, y, W):
    # r*(x,y) = <W_y, x>
    return torch.einsum('yd,bd->by', W, x)[torch.arange(x.size(0)), y]

def generate_dataset(
    n_pairs=200_000,
    n_val_calib=10_000,
    n_test=50_000,
    n_actions=10,
    feat_dim=10,
    link='beta_sigmoid',
    beta_a=2.0, beta_b=5.0,
    seed=123,
    device='cuda'
):
    set_seed(seed)
    W = make_reward_weights(n_actions, feat_dim, device)

    def sample_feats(n):
        return torch.randn(n, feat_dim, device=device)

    # Pair sampling
    def sample_pairs(n):
        x = sample_feats(n)
        y0 = torch.randint(0, n_actions, (n,), device=device)
        y1 = torch.randint(0, n_actions, (n,), device=device)
        # prevent equal pairs
        mask = (y1 == y0)
        while mask.any():
            y1[mask] = torch.randint(0, n_actions, (mask.sum().item(),), device=device)
            mask = (y1 == y0)
        # reward diffs
        r0 = true_reward(x, y0, W)
        r1 = true_reward(x, y1, W)
        delta = r1 - r0
        if link == 'beta_sigmoid':
            p = beta_sigmoid_cdf(delta, a=beta_a, b=beta_b)
        else:
            raise ValueError(f"Unknown link: {link}")
        z = torch.bernoulli(p).long()
        return x, y0, y1, z, W

    train = sample_pairs(n_pairs)
    calib_x = torch.randn(n_val_calib, feat_dim, device=device)
    test_x = torch.randn(n_test, feat_dim, device=device)

    return {
        'train': train,      # (x, y0, y1, z, W)
        'calib_x': calib_x,  # used for KL calibration
        'test_x': test_x,    # used for evaluation
        'W': train[-1]
    }

