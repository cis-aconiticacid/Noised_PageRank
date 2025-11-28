"""Helpers to run trajectories and compute Hilbert diagnostics."""

from __future__ import annotations

from typing import Callable, Iterable

import numpy as np

from ..cones.hilbert_metric import hilbert_ratio_to_final, hilbert_ratios_between

TrajectoryResult = dict[str, np.ndarray]


def run_trajectory(
    iterate_fn: Callable[..., np.ndarray],
    hilbert: bool = True,
    x_final: np.ndarray | None = None,
    **kwargs,
) -> TrajectoryResult:
    """Run a trajectory-producing function and optionally compute Hilbert distances."""
    traj = iterate_fn(**kwargs)
    result: TrajectoryResult = {"traj": traj}
    if hilbert and x_final is not None:
        result["hilbert_to_final"] = hilbert_ratio_to_final(traj, final_idx=-1)
    if hilbert:
        result["hilbert_between"] = hilbert_ratios_between(traj)
    return result


def compute_hilbert_diagnostics(traj: Iterable[np.ndarray], x_final: np.ndarray) -> TrajectoryResult:
    """Compute Hilbert to-final and between-step distances for a trajectory."""
    trajectory = list(traj)
    return {
        "hilbert_to_final": hilbert_ratio_to_final(trajectory, final_idx=-1),
        "hilbert_between": hilbert_ratios_between(trajectory),
    }

