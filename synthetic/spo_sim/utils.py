import math
import os
import random
import time
from dataclasses import dataclass
from typing import Iterable, List, Tuple

import numpy as np
import torch


@dataclass
class TrainConfig:
    batch_size: int = 128
    epochs: int = 50
    lr: float = 1e-3
    weight_decay: float = 0.0
    device: str = "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer_cuda: bool = True) -> str:
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def chunks(n: int, batch_size: int) -> Iterable[Tuple[int, int]]:
    for start in range(0, n, batch_size):
        end = min(n, start + batch_size)
        yield start, end


def logit(p: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    p = torch.clamp(p, eps, 1 - eps)
    return torch.log(p) - torch.log1p(-p)


def softplus(x: torch.Tensor) -> torch.Tensor:
    return torch.nn.functional.softplus(x)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def now() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


class Timer:
    def __init__(self):
        self.start = time.time()

    @property
    def elapsed(self) -> float:
        return time.time() - self.start
