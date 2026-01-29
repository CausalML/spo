from __future__ import annotations

import os
import random
from typing import Optional

import numpy as np

try:
    import torch
except Exception:  # pragma: no cover - torch optional for small tools
    torch = None


def seed_everything(seed: int, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False


def get_worker_seed(base_seed: int, worker_id: int) -> int:
    return (base_seed + worker_id * 100003) % (2**32 - 1)
