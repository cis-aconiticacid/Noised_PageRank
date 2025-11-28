"""PageRank dynamics with entrywise thresholding/sparsification."""

from __future__ import annotations

import numpy as np

from ..cones.cone_utils import normalize_prob


def apply_threshold(x: np.ndarray, tau: float) -> np.ndarray:
    """Zero out entries of ``x`` that fall below ``tau``."""
    masked = np.asarray(x, dtype=float).copy()
    masked[masked < tau] = 0.0
    return masked


def iterate_thresholded_pagerank(M: np.ndarray, x0: np.ndarray, num_steps: int, tau: float) -> np.ndarray:
    """Iterate PageRank with hard thresholding after each multiplication."""
    x = normalize_prob(x0)
    traj = [x]
    for _ in range(num_steps):
        z = apply_threshold(M @ x, tau=tau)
        x = normalize_prob(z)
        traj.append(x)
    return np.asarray(traj)

