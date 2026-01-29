from __future__ import annotations

from typing import List, Tuple

import numpy as np


def pava(x: np.ndarray, y: np.ndarray, eps: float = 1e-6) -> Tuple[np.ndarray, np.ndarray]:
    """Pool Adjacent Violators Algorithm for isotonic regression.

    Returns sorted unique x and fitted nondecreasing y_hat.
    """
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]

    # Initialize blocks
    blocks = [(x_sorted[i], x_sorted[i], y_sorted[i], 1) for i in range(len(x_sorted))]

    def merge(b1, b2):
        x_min = b1[0]
        x_max = b2[1]
        weight = b1[3] + b2[3]
        y_avg = (b1[2] * b1[3] + b2[2] * b2[3]) / weight
        return (x_min, x_max, y_avg, weight)

    i = 0
    while i < len(blocks) - 1:
        if blocks[i][2] <= blocks[i + 1][2] + eps:
            i += 1
            continue
        merged = merge(blocks[i], blocks[i + 1])
        blocks[i : i + 2] = [merged]
        i = max(i - 1, 0)

    x_hat = np.array([0.5 * (b[0] + b[1]) for b in blocks])
    y_hat = np.array([b[2] for b in blocks])
    return x_hat, y_hat


def isotonic_predict(x_query: np.ndarray, x_hat: np.ndarray, y_hat: np.ndarray) -> np.ndarray:
    """Piecewise-constant isotonic predictor."""
    idx = np.searchsorted(x_hat, x_query, side="right") - 1
    idx = np.clip(idx, 0, len(y_hat) - 1)
    return y_hat[idx]
