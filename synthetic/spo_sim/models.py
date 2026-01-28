from typing import Tuple

import torch
from torch import nn


class MLPPolicy(nn.Module):
    def __init__(self, d: int, k: int, hidden_sizes: Tuple[int, ...]):
        super().__init__()
        layers = []
        in_dim = d
        for h in hidden_sizes:
            layers.append(nn.Linear(in_dim, h))
            layers.append(nn.ReLU())
            in_dim = h
        layers.append(nn.Linear(in_dim, k))
        self.net = nn.Sequential(*layers)
        self.k = k

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def log_probs(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.log_softmax(logits, dim=-1)

    def probs(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.forward(x)
        return torch.softmax(logits, dim=-1)
