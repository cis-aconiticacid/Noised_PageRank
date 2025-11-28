"""PageRank dynamics perturbed by additive noise."""

from __future__ import annotations

import numpy as np

from ..cones.cone_utils import normalize_prob, project_to_positive_cone


def sample_noise(shape: tuple[int, ...], sigma: float, rng: np.random.Generator | None = None) -> np.ndarray:
    """Sample zero-mean Gaussian noise with scale ``sigma``."""
    generator = rng if rng is not None else np.random.default_rng()
    return generator.normal(loc=0.0, scale=sigma, size=shape)


def iterate_noisy_pagerank(
    M: np.ndarray,
    x0: np.ndarray,
    num_steps: int,
    sigma: float,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Iterate PageRank with additive noise followed by projection and normalization."""
    x = normalize_prob(x0)
    traj = [x]
    generator = rng if rng is not None else np.random.default_rng()
    for _ in range(num_steps):
        noise = sample_noise(x.shape, sigma=sigma, rng=generator)
        z = project_to_positive_cone(M @ x + noise)
        x = normalize_prob(z)
        traj.append(x)
    return np.asarray(traj)

