"""Utilities for working with the positive cone and probability vectors."""

from __future__ import annotations

import numpy as np


def is_in_positive_cone(x: np.ndarray, eps: float = 0.0) -> bool:
    """Return ``True`` when every entry of ``x`` is at least ``eps``."""
    array = np.asarray(x)
    return bool(np.all(array >= eps))


def project_to_positive_cone(x: np.ndarray, clip_eps: float = 0.0) -> np.ndarray:
    """Clip entries of ``x`` to make them nonnegative.

    Parameters
    ----------
    x:
        Input vector.
    clip_eps:
        Minimum value enforced on every entry.
    """
    return np.clip(np.asarray(x, dtype=float), clip_eps, None)


def normalize_prob(x: np.ndarray, eps: float = 1e-18) -> np.ndarray:
    """Normalize ``x`` to a probability vector.

    Negative entries are clipped to zero before normalization. If the vector is
    too close to the zero vector, ``eps`` is added uniformly to avoid division
    by zero.
    """
    clipped = project_to_positive_cone(x)
    total = float(clipped.sum())
    if total < eps:
        clipped = clipped + eps
        total = float(clipped.sum())
    return clipped / total


def random_interior_point(n: int, low: float = 0.1, high: float = 1.0, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample a strictly positive probability vector of dimension ``n``.

    The entries are drawn uniformly from ``[low, high]`` and then normalized.
    """
    generator = rng if rng is not None else np.random.default_rng()
    raw = generator.uniform(low=low, high=high, size=n)
    return normalize_prob(raw)

