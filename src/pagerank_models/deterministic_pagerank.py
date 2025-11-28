"""Deterministic PageRank dynamics and utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from ..cones.cone_utils import normalize_prob, random_interior_point


def build_google_matrix(n_nodes: int, alpha: float = 0.85, rng: np.random.Generator | None = None) -> np.ndarray:
    """Construct a column-stochastic Google matrix with teleportation.

    The base transition matrix is generated randomly and column-normalized.
    Teleportation is added using a strictly positive vector to guarantee
    interior mapping.
    """
    generator = rng if rng is not None else np.random.default_rng()
    base = generator.random((n_nodes, n_nodes))
    column_sums = base.sum(axis=0, keepdims=True)
    stochastic = base / column_sums

    teleport = random_interior_point(n_nodes, rng=generator)
    teleport_matrix = teleport[:, None]
    return alpha * stochastic + (1.0 - alpha) * teleport_matrix


def iterate_pagerank(M: np.ndarray, x0: np.ndarray, num_steps: int) -> np.ndarray:
    """Iterate the PageRank map, returning the full trajectory (including ``x0``)."""
    x = normalize_prob(x0)
    traj = [x]
    for _ in range(num_steps):
        x = normalize_prob(M @ x)
        traj.append(x)
    return np.asarray(traj)


def compute_pagerank_steady_state(M: np.ndarray, tol: float = 1e-12, max_iters: int = 10000) -> np.ndarray:
    """Approximate the stationary distribution via power iteration."""
    n = M.shape[0]
    x = random_interior_point(n)
    for _ in range(max_iters):
        x_next = normalize_prob(M @ x)
        if np.linalg.norm(x_next - x, ord=1) < tol:
            break
        x = x_next
    return x


def iterate_multiple(M: np.ndarray, initial_points: Iterable[np.ndarray], num_steps: int) -> list[np.ndarray]:
    """Iterate PageRank from several starting points."""
    return [iterate_pagerank(M, x0, num_steps=num_steps) for x0 in initial_points]

