"""High-level helpers to compare NN and PageRank trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from ..cones.hilbert_metric import hilbert_ratio_to_final, hilbert_ratios_between
from .trajectory_loader import trajectory_to_cone


@dataclass
class ContractionSummary:
    hilbert_to_final: np.ndarray
    hilbert_between: np.ndarray


def summarize_trajectory(traj: Iterable[np.ndarray], final_idx: int = -1) -> ContractionSummary:
    """Compute Hilbert-based summaries for a trajectory."""
    trajectory = np.asarray(list(traj), dtype=float)
    return ContractionSummary(
        hilbert_to_final=hilbert_ratio_to_final(trajectory, final_idx=final_idx),
        hilbert_between=hilbert_ratios_between(trajectory),
    )


def compare_contraction_rates(
    nn_traj: np.ndarray,
    pr_traj: np.ndarray,
    nn_final_idx: int = -1,
    pr_final_idx: int = -1,
    project_nn: bool = True,
) -> dict[str, ContractionSummary]:
    """Return Hilbert summaries for NN and PageRank trajectories."""
    nn_processed = trajectory_to_cone(nn_traj) if project_nn else np.asarray(nn_traj, dtype=float)
    pr_processed = np.asarray(pr_traj, dtype=float)
    return {
        "nn": summarize_trajectory(nn_processed, final_idx=nn_final_idx),
        "pagerank": summarize_trajectory(pr_processed, final_idx=pr_final_idx),
    }

