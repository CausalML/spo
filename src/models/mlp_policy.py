import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPPolicy(nn.Module):
    def __init__(self, input_dim=10, n_actions=10, hidden_sizes=(128,128,64), activation='gelu'):
        super().__init__()
        act = {'relu': nn.ReLU, 'gelu': nn.GELU, 'tanh': nn.Tanh}.get(activation, nn.GELU)
        hs = list(hidden_sizes)
        layers = []
        prev = input_dim
        for h in hs:
            layers += [nn.Linear(prev, h), act()]
            prev = h
        self.body = nn.Sequential(*layers)
        self.head = nn.Linear(prev, n_actions)

    def logits(self, x):   # logits
        return self.head(self.body(x))

    def forward(self, x):   # log πθ(y|x)
        logits = self.head(self.body(x))
        return logits.log_softmax(dim=-1)

    def logprobs(self, x):  # also log πθ(y|x)
        return self.forward(x)

    def probs(self, x):     # πθ(y|x)
        return self.forward(x).exp()
