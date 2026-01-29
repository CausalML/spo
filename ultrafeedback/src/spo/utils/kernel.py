from __future__ import annotations

from typing import Callable

import numpy as np


def gaussian_kernel(u: np.ndarray) -> np.ndarray:
    return np.exp(-0.5 * u * u)


def epanechnikov_kernel(u: np.ndarray) -> np.ndarray:
    mask = np.abs(u) <= 1
    out = np.zeros_like(u)
    out[mask] = 0.75 * (1 - u[mask] ** 2)
    return out


def get_kernel(name: str) -> Callable[[np.ndarray], np.ndarray]:
    if name == "gaussian":
        return gaussian_kernel
    if name == "epanechnikov":
        return epanechnikov_kernel
    raise ValueError(f"Unknown kernel: {name}")
