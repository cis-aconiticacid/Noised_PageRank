# Hilbert-Metric Dynamics of PageRank and Noisy Cone Systems

This repository hosts a small, self-contained toolkit for experimenting with PageRank dynamics inside the positive cone and inspecting their behavior under Hilbert's projective metric. The code is organized to support three core goals:

1. Reproduce the classical Hilbert contraction of clean PageRank iterations.
2. Introduce thresholding or sparsification to study boundary-hitting effects.
3. Add noise to draw analogies between PageRank dynamics and stochastic optimization, and compare against neural-network trajectories.

## Layout

```
Noised_PageRank/
├── src/
│   ├── cones/                  # Hilbert metric and cone utilities
│   ├── pagerank_models/        # Deterministic, thresholded, and noisy PageRank maps
│   ├── experiments/            # Trajectory runners, diagnostics, and plotting helpers
│   └── nn_bridge/              # Utilities for loading NN trajectories and comparing them
└── notebooks/                  # Experiment orchestration and figure notebooks
```

Each submodule is written to be composable: the `cones` utilities can be reused in other cone-based projects, while `pagerank_models` exposes small, explicit iteration functions for experimentation.

## Getting started

Install the minimal dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then try the setup notebook to verify your environment:

```bash
jupyter notebook notebooks/00_setup_and_sanity_checks.ipynb
```

## Reproducibility

All random operations use NumPy's `Generator` API. Pass in an explicit `rng` to any helper that accepts one to ensure replicable trajectories.
