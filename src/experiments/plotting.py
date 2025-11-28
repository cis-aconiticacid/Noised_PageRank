"""Plotting utilities for Hilbert-distance diagnostics."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def _finalize(fig, ax, title: str, xlabel: str, ylabel: str, save_path: str | None) -> None:
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)


def plot_hilbert_to_final(values: np.ndarray, title: str = "Hilbert distance to final", save_path: str | None = None) -> None:
    fig, ax = plt.subplots()
    ax.plot(values, label="d_H(x_t, x_final)")
    ax.legend()
    _finalize(fig, ax, title=title, xlabel="iteration", ylabel="distance", save_path=save_path)


def plot_hilbert_between(values: np.ndarray, title: str = "Hilbert distance between steps", save_path: str | None = None) -> None:
    fig, ax = plt.subplots()
    ax.plot(values, label="d_H(x_{t+1}, x_t)")
    ax.legend()
    _finalize(fig, ax, title=title, xlabel="iteration", ylabel="distance", save_path=save_path)


def plot_both(to_final: np.ndarray, between: np.ndarray, title: str = "Hilbert metrics", save_path: str | None = None) -> None:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(to_final, label="to final")
    ax1.legend()
    ax2.plot(between, label="between")
    ax2.legend()
    _finalize(fig, ax1, title=title, xlabel="iteration", ylabel="distance", save_path=None)
    ax2.set_title("Between steps")
    ax2.set_xlabel("iteration")
    ax2.set_ylabel("distance")
    fig.tight_layout()
    if save_path is not None:
        fig.savefig(save_path)


def plot_trajectory_simplex(traj: np.ndarray, dims: tuple[int, int, int] = (0, 1, 2), title: str = "Simplex trajectory", save_path: str | None = None) -> None:
    if traj.shape[1] < 3:
        raise ValueError("Simplex plotting requires at least 3 dimensions.")
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    coords = traj[:, list(dims)]
    ax.plot(coords[:, 0], coords[:, 1], coords[:, 2])
    ax.set_title(title)
    ax.set_xlabel(f"dim {dims[0]}")
    ax.set_ylabel(f"dim {dims[1]}")
    ax.set_zlabel(f"dim {dims[2]}")
    if save_path is not None:
        fig.savefig(save_path)
    fig.tight_layout()

