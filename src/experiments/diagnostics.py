"""Diagnostic helpers for Hilbert-distance trajectories."""

from __future__ import annotations

import numpy as np


def compute_basic_stats(values: np.ndarray) -> dict[str, float]:
    """Return simple summary statistics for a 1D array."""
    array = np.asarray(values, dtype=float)
    return {
        "initial": float(array[0]) if array.size else float("nan"),
        "final": float(array[-1]) if array.size else float("nan"),
        "min": float(array.min()) if array.size else float("nan"),
        "max": float(array.max()) if array.size else float("nan"),
    }


def find_spikes(values: np.ndarray, threshold: float) -> np.ndarray:
    """Return indices where ``values`` exceed ``threshold``."""
    array = np.asarray(values, dtype=float)
    return np.nonzero(array > threshold)[0]


def estimate_contraction_ratio(values: np.ndarray) -> float:
    """Estimate an average contraction ratio from successive values."""
    array = np.asarray(values, dtype=float)
    if array.size < 2:
        return float("nan")
    deltas = np.diff(array)
    average_step = np.mean(np.abs(deltas))
    baseline = np.mean(array[:-1])
    if baseline == 0:
        return float("nan")
    return float(1.0 - average_step / baseline)

