from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np


def logistic(u: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-u))


def shifted_logistic_mixture(u: np.ndarray, shift: float, temperature: float) -> np.ndarray:
    u_scaled = u / max(temperature, 1e-8)
    return 0.5 * logistic(u_scaled - shift) + 0.5 * logistic(u_scaled + shift)


def get_link(link_type: str, shift: float, temperature: float) -> Callable[[np.ndarray], np.ndarray]:
    if link_type == "shifted_logistic_mixture":
        return lambda u: shifted_logistic_mixture(u, shift=shift, temperature=temperature)
    raise ValueError(f"Unknown link_type: {link_type}")


def normalize_reward_diffs(diffs: np.ndarray, target_std: float = 1.0) -> Tuple[np.ndarray, float]:
    std = np.std(diffs)
    if std <= 1e-8:
        return diffs, 1.0
    scale = target_std / std
    return diffs * scale, scale


def sample_preferences(
    reward_diffs: np.ndarray,
    link_type: str,
    shift: float,
    temperature: float,
    rng: np.random.Generator,
) -> np.ndarray:
    link = get_link(link_type, shift=shift, temperature=temperature)
    probs = link(reward_diffs)
    return rng.binomial(1, probs).astype(np.int32)
