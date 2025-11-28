"""Utilities for loading neural network trajectories and projecting them into the cone."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

from ..cones.cone_utils import normalize_prob, project_to_positive_cone


def _load_torch_tensor(path: Path) -> np.ndarray:
    if importlib.util.find_spec("torch") is None:
        raise ImportError("PyTorch is required to load .pt trajectories.")
    import torch

    tensor = torch.load(path, map_location="cpu")
    array = tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor)
    return np.asarray(array, dtype=float)


def load_nn_trajectory_np(path: str | Path) -> np.ndarray:
    """Load a numpy or torch trajectory file as a float array."""
    path = Path(path)
    if path.suffix in {".pt", ".pth"}:
        return _load_torch_tensor(path)
    return np.asarray(np.load(path), dtype=float)


def to_probability_vector(x: np.ndarray) -> np.ndarray:
    """Map an arbitrary vector to the positive cone probability simplex."""
    return normalize_prob(project_to_positive_cone(x, clip_eps=0.0))


def trajectory_to_cone(traj: np.ndarray) -> np.ndarray:
    """Project an entire trajectory into the probability simplex."""
    return np.asarray([to_probability_vector(x) for x in traj])

