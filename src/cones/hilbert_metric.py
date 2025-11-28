"""Hilbert metric utilities on the positive orthant."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def check_interior(x: np.ndarray, eps: float = 1e-12) -> bool:
    """Return ``True`` when ``x`` lies in the numerical interior of the cone."""
    array = np.asarray(x)
    return bool(np.all(array >= eps))


def _validate_same_shape(x: np.ndarray, y: np.ndarray) -> None:
    if x.shape != y.shape:
        raise ValueError(f"Shape mismatch: {x.shape} vs {y.shape}")


def hilbert_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Compute Hilbert's projective metric between two interior points.

    Parameters
    ----------
    x, y:
        Vectors with strictly positive entries (up to ``eps``).
    eps:
        Numerical tolerance for interior checks.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    _validate_same_shape(x_arr, y_arr)
    if not (check_interior(x_arr, eps) and check_interior(y_arr, eps)):
        raise ValueError("Hilbert distance is only defined for interior points of the cone.")

    ratios = x_arr / y_arr
    max_ratio = ratios.max()
    min_ratio = ratios.min()
    return float(math.log(max_ratio / min_ratio))


def safe_hilbert_distance(x: np.ndarray, y: np.ndarray, eps: float = 1e-12) -> float:
    """Hilbert distance that returns ``np.inf`` if either point is outside the interior."""
    try:
        return hilbert_distance(x, y, eps=eps)
    except ValueError:
        return math.inf


def hilbert_ratio_to_final(traj: Iterable[np.ndarray], final_idx: int = -1, eps: float = 1e-12) -> np.ndarray:
    """Compute Hilbert distances from each point in ``traj`` to the final point.

    Parameters
    ----------
    traj:
        Sequence of vectors ``[x_0, x_1, ..., x_T]``.
    final_idx:
        Index of the reference point. By default, the last element is used.
    eps:
        Numerical tolerance for interior checks.
    """
    trajectory = np.asarray(list(traj), dtype=float)
    if trajectory.ndim < 2:
        raise ValueError("Trajectory must be a sequence of vectors.")
    reference = trajectory[final_idx]
    distances = [hilbert_distance(x, reference, eps=eps) for x in trajectory]
    return np.asarray(distances, dtype=float)


def hilbert_ratios_between(traj: Iterable[np.ndarray], eps: float = 1e-12) -> np.ndarray:
    """Compute Hilbert distances between consecutive points in ``traj``."""
    trajectory = np.asarray(list(traj), dtype=float)
    if trajectory.ndim < 2:
        raise ValueError("Trajectory must be a sequence of vectors.")
    if len(trajectory) < 2:
        return np.array([], dtype=float)
    return np.asarray(
        [hilbert_distance(trajectory[i + 1], trajectory[i], eps=eps) for i in range(len(trajectory) - 1)],
        dtype=float,
    )

