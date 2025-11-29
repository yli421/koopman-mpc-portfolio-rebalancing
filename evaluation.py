"""Comprehensive evaluation utilities for Koopman Autoencoder models.

This module implements the evaluation protocol described in the research
specification. It supports multiple rollout strategies, computes horizon-wise
mean-squared error metrics, and produces qualitative plots such as phase
portraits and MSE-vs-horizon curves.

Key features
------------
- Rollout generators for:
  * No reencoding (latent-only evolution)
  * Every-step reencoding (state-space evolution)
  * Periodic reencoding with configurable period k
- Evaluation over multiple dynamical systems, horizons, and reencoding periods
- Aggregation of metrics across unseen initial conditions (mean ± std)
- Automatic selection of the best periodic reencoding period per horizon
- Qualitative plots with ground truth trajectories in transparent gray

The public entry point is :func:`evaluate_model`, which returns a nested metrics
dictionary and optionally saves metrics/plots to disk.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from config import Config
from data import VectorWrapper, generate_trajectory, make_env
from model import KoopmanMachine
from data import LyapunovMultiAttractor, make_env


# ---------------------------------------------------------------------------
# Rollout generators
# ---------------------------------------------------------------------------


@torch.no_grad()
def rollout_no_reencode(model: KoopmanMachine, x0: torch.Tensor, horizon: int) -> torch.Tensor:
    """Roll out the Koopman dynamics without reencoding.

    Args:
        model: Trained Koopman machine.
        x0: Initial states with shape ``[batch, state_dim]``.
        horizon: Number of prediction steps.

    Returns:
        Predicted trajectory with shape ``[horizon, batch, state_dim]``.
    """
    model.eval()
    device = next(model.parameters()).device
    x0 = x0.to(device)

    latent = model.encode(x0)
    predictions: List[torch.Tensor] = []

    for _ in range(horizon):
        latent = model.step_latent(latent)
        x_pred = model.decode(latent)
        predictions.append(x_pred)

        if not torch.isfinite(x_pred).all():
            # Mark remaining steps as NaN to signal explosion
            nan_frame = torch.full_like(x_pred, torch.nan)
            predictions.extend([nan_frame] * (horizon - len(predictions)))
            break

    return torch.stack(predictions, dim=0)


@torch.no_grad()
def rollout_every_step_reencode(
    model: KoopmanMachine,
    x0: torch.Tensor,
    horizon: int,
) -> torch.Tensor:
    """Roll out the Koopman dynamics with reencoding at every step."""

    model.eval()
    device = next(model.parameters()).device
    state = x0.to(device)
    predictions: List[torch.Tensor] = []

    for _ in range(horizon):
        state = model.step_env(state)
        predictions.append(state)

        if not torch.isfinite(state).all():
            nan_frame = torch.full_like(state, torch.nan)
            predictions.extend([nan_frame] * (horizon - len(predictions)))
            break

    return torch.stack(predictions, dim=0)


@torch.no_grad()
def rollout_periodic_reencode(
    model: KoopmanMachine,
    x0: torch.Tensor,
    horizon: int,
    period: int,
) -> torch.Tensor:
    """Roll out the Koopman dynamics with periodic reencoding every *period* steps."""

    if period <= 0:
        raise ValueError("period must be a positive integer")

    model.eval()
    device = next(model.parameters()).device
    x0 = x0.to(device)

    latent = model.encode(x0)
    predictions: List[torch.Tensor] = []

    for step in range(horizon):
        latent = model.step_latent(latent)
        x_pred = model.decode(latent)
        predictions.append(x_pred)

        if not torch.isfinite(x_pred).all():
            nan_frame = torch.full_like(x_pred, torch.nan)
            predictions.extend([nan_frame] * (horizon - len(predictions)))
            break

        if (step + 1) % period == 0:
            latent = model.encode(x_pred)

    return torch.stack(predictions, dim=0)


# ---------------------------------------------------------------------------
# Metric computation helpers
# ---------------------------------------------------------------------------


def _compute_horizon_mse(
    squared_errors: torch.Tensor,
    horizon: int,
) -> Tuple[float, float, List[float], int]:
    """Compute mean ± std MSE for a specific horizon.

    Args:
        squared_errors: Tensor ``[time, batch]`` with per-step squared L2 norms.
        horizon: Horizon length (<= time dimension of squared_errors).

    Returns:
        Tuple ``(mean, std, per_ic, num_valid)`` where *per_ic* is a list of the
        per-initial-condition MSE values used for aggregation.
    """

    horizon = min(horizon, squared_errors.size(0))
    horizon_errors = squared_errors[:horizon]

    # Average over time, ignoring NaNs (exploding rollouts)
    per_ic = torch.nanmean(horizon_errors, dim=0)
    valid_mask = torch.isfinite(per_ic)

    if valid_mask.sum() == 0:
        return float("nan"), float("nan"), [], 0

    valid_errors = per_ic[valid_mask]
    mean = valid_errors.mean().item()
    std = valid_errors.std(unbiased=False).item() if valid_errors.numel() > 1 else 0.0
    return mean, std, valid_errors.tolist(), int(valid_mask.sum().item())


def _cumulative_mse_curve(squared_errors: torch.Tensor) -> List[float]:
    """Compute cumulative MSE curve averaged across initial conditions."""

    time_steps = squared_errors.size(0)
    steps = torch.arange(1, time_steps + 1, dtype=torch.float32, device=squared_errors.device)
    cumulative = torch.cumsum(squared_errors, dim=0)
    with torch.no_grad():
        curve = torch.nanmean(cumulative / steps.view(-1, 1), dim=1)
    return curve.cpu().tolist()


# ---------------------------------------------------------------------------
# Plotting utilities
# ---------------------------------------------------------------------------


def _ensure_matplotlib():
    import matplotlib

    if matplotlib.get_backend().lower() != "agg":
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: F401  # pylint: disable=unused-import


def _save_phase_portrait_overlay(
    true_sequences: torch.Tensor,
    predicted_sequences: Dict[str, torch.Tensor],
    path: Path,
    max_samples: int = 20,
) -> None:
    """Save a phase portrait overlay plot.

    Args:
        true_sequences: Tensor with shape ``[batch, time + 1, state_dim]``.
        predicted_sequences: Mapping from mode name to tensor with shape
            ``[batch, time, state_dim]``.
        path: Output path for the PNG file.
        max_samples: Maximum number of trajectories to render.
    """

    if true_sequences.size(-1) < 2:
        return  # Phase portrait not meaningful

    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors

    path.parent.mkdir(parents=True, exist_ok=True)

    batch = true_sequences.size(0)
    indices = torch.arange(batch)

    # Filter trajectories with finite predictions for all modes
    finite_mask = torch.ones(batch, dtype=torch.bool)
    true_xy = true_sequences[:, :, :2]

    for preds in predicted_sequences.values():
        flat = preds.reshape(preds.size(0), -1)
        finite_mask &= torch.isfinite(flat).all(dim=1)

    indices = indices[finite_mask]
    if indices.numel() == 0:
        return

    indices = indices[:max_samples]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Plot ground truth trajectories in light gray
    for idx in indices.tolist():
        gt = true_xy[idx].cpu().numpy()
        ax.plot(gt[:, 0], gt[:, 1], color=(0.5, 0.5, 0.5), alpha=0.25, linewidth=1.5)

    rng = np.random.default_rng(42)
    colors = {
        mode: mcolors.hsv_to_rgb([float(rng.random()), 0.65 + 0.3 * float(rng.random()), 0.9])
        for mode in predicted_sequences.keys()
    }

    for mode, preds in predicted_sequences.items():
        color = colors[mode]
        for idx in indices.tolist():
            pred_xy = torch.cat([
                true_xy[idx, :1],
                preds[idx, :, :2],
            ], dim=0).cpu().numpy()
            ax.plot(
                pred_xy[:, 0],
                pred_xy[:, 1],
                color=color,
                alpha=0.9,
                linewidth=1.2,
                label=mode if idx == indices[0].item() else None,
            )

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title("Phase portrait (1000-step)")
    ax.set_xlim(-2.5, 2.5)
    ax.set_ylim(-2.5, 2.5)
    ax.set_aspect("equal", adjustable="box")
    # ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_phase_portrait_single_mode(
    true_sequences: torch.Tensor,
    predicted: torch.Tensor,
    path: Path,
    max_samples: int = 20,
    title: Optional[str] = None,
    axis_lim: float = 2.5,
) -> None:
    """Save a phase portrait for a single rollout mode, coloring each trajectory.

    Args:
        true_sequences: Tensor with shape ``[batch, time + 1, state_dim]``.
        predicted: Tensor with shape ``[batch, time, state_dim]`` for the mode.
        path: Output PNG path.
        max_samples: Maximum number of trajectories to render.
        title: Optional plot title.
        axis_lim: Axis limits for the plot (default 2.5).
    """

    if true_sequences.size(-1) < 2:
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import matplotlib.cm as cm

    path.parent.mkdir(parents=True, exist_ok=True)

    batch = true_sequences.size(0)
    true_xy = true_sequences[:, :, :2]

    # Keep only trajectories with finite predictions
    flat = predicted.reshape(predicted.size(0), -1)
    finite_mask = torch.isfinite(flat).all(dim=1)
    indices = torch.arange(batch)[finite_mask]
    if indices.numel() == 0:
        return
    indices = indices[:max_samples]

    fig, ax = plt.subplots(1, 1, figsize=(7, 6))

    # Colormap per-trajectory
    cmap = cm.get_cmap("tab20", indices.numel())

    for j, idx in enumerate(indices.tolist()):
        # predicted (plot first, underneath)
        pred_xy = torch.cat([true_xy[idx, :1], predicted[idx, :, :2]], dim=0).cpu().numpy()
        ax.plot(pred_xy[:, 0], pred_xy[:, 1], color=cmap(j), linewidth=1.5, zorder=2)

        # ground truth in light gray (plot last, on top)
        gt = true_xy[idx].cpu().numpy()
        ax.plot(gt[:, 0], gt[:, 1], color=(0.6, 0.6, 0.6), alpha=0.5, linewidth=1.5, zorder=3)

    ax.set_xlabel("x1")
    ax.set_ylabel("x2")
    ax.set_title(title or "Phase portrait (single mode)")
    ax.set_xlim(-axis_lim, axis_lim)
    ax.set_ylim(-axis_lim, axis_lim)
    ax.set_aspect("equal", adjustable="box")
    # ax.set_aspect("equal", adjustable="datalim")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)

def _save_mse_curve_plot(curves: Dict[str, List[float]], path: Path, highlight_horizons: Sequence[int]) -> None:
    """Save MSE vs horizon curves for each rollout mode."""

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for mode, curve in curves.items():
        xs = np.arange(1, len(curve) + 1)
        ax.plot(xs, curve, linewidth=2, label=mode)

    for horizon in highlight_horizons:
        ax.axvline(horizon, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Prediction horizon")
    ax.set_ylabel("Mean MSE")
    ax.set_title("MSE vs horizon")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_error_curve_single_mode(
    errors: torch.Tensor,
    path: Path,
    title: Optional[str] = None,
) -> None:
    """Save per-timestep mean L2 error for a single rollout mode."""

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    error_np = errors.cpu().numpy()
    steps = np.arange(1, error_np.shape[0] + 1)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    ax.plot(steps, error_np, linewidth=2)
    ax.set_xlabel("Prediction step")
    ax.set_ylabel("Mean L2 error")
    ax.set_title(title or "Per-step prediction error")
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _save_error_curve_combined(
    errors_by_mode: Dict[str, torch.Tensor],
    path: Path,
    highlight_steps: Optional[Sequence[int]] = None,
) -> None:
    """Save combined per-step mean error curves for all rollout modes."""

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(7, 5))
    for mode, errors in errors_by_mode.items():
        error_np = errors.cpu().numpy()
        steps = np.arange(1, error_np.shape[0] + 1)
        ax.plot(steps, error_np, linewidth=2, label=mode)

    if highlight_steps is not None:
        for step in highlight_steps:
            if step <= 0:
                continue
            ax.axvline(step, color="gray", linestyle="--", linewidth=1.0, alpha=0.5)

    ax.set_xlabel("Prediction step")
    ax.set_ylabel("Mean L2 error")
    ax.set_title("Per-step prediction error (all modes)")
    ax.grid(True, linestyle=":", alpha=0.4)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def _save_vector_magnitude_histogram(
    magnitudes: np.ndarray,
    path: Path,
    title: str,
    bins: int = 30,
) -> None:
    """Save a histogram of vector magnitudes used in a phase portrait."""

    flat = np.asarray(magnitudes, dtype=np.float32).ravel()
    if flat.size == 0:
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(1, 1, figsize=(6, 4))
    ax.hist(flat, bins=bins, color="#4682B4", alpha=0.85, edgecolor="white")
    ax.set_xlabel("Vector magnitude")
    ax.set_ylabel("Count")
    ax.set_title(title)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _estimate_learned_attractors(
    model: KoopmanMachine,
    grid_lim: float,
    num_samples: int,
    num_steps: int,
    tolerance: float,
    device: torch.device,
    seed: int = 7,
) -> np.ndarray:
    """Estimate attractor locations of the learned system via rollouts."""

    rng = np.random.default_rng(seed)
    samples = rng.uniform(-grid_lim, grid_lim, size=(num_samples, 2)).astype(np.float32)
    attractors: List[np.ndarray] = []

    print(
        f"[lyapunov] Estimating learned attractors (samples={num_samples}, "
        f"steps={num_steps}, tolerance={tolerance})",
        flush=True,
    )

    report_interval = max(1, num_samples // 5)
    for idx, sample in enumerate(samples):
        state = torch.from_numpy(sample).to(device)
        with torch.no_grad():
            for _ in range(num_steps):
                state = model.step_env(state.unsqueeze(0)).squeeze(0)
        final_state = state.cpu().numpy()

        # NOTE: check this? we are just making the first final 
        # state an attractor
        if not attractors:
            attractors.append(final_state)
            continue

        existing = np.asarray(attractors)
        dists = np.linalg.norm(existing - final_state, axis=1)
        if float(dists.min()) > tolerance:
            attractors.append(final_state)

        if (idx + 1) % report_interval == 0 or (idx + 1) == num_samples:
            print(
                f"[lyapunov]   processed {idx + 1}/{num_samples} samples "
                f"(unique attractors={len(attractors)})",
                flush=True,
            )

    if not attractors:
        print("[lyapunov] No attractors detected; returning empty array.", flush=True)
        return np.empty((0, samples.shape[1]), dtype=np.float32)

    stacked = np.stack(attractors, axis=0)
    print(
        f"[lyapunov] Attractor estimation complete: {len(attractors)} unique points.",
        flush=True,
    )
    return stacked


def _save_lyapunov_phase_portrait_comparison(
    model: KoopmanMachine,
    env: LyapunovMultiAttractor,
    path: Path,
    num_trajectories: int = 12,
    grid_lim: float = 3.0,
    grid_n: int = 15,
) -> Dict[str, str]:
    """Notebook-style comparison plot for the Lyapunov system with extras."""

    print("[lyapunov] Preparing phase portrait comparison...", flush=True)

    # Lazy imports for plotting and optional Voronoi
    _ensure_matplotlib()
    import matplotlib.pyplot as plt
    import numpy as np

    try:
        from scipy.spatial import Voronoi  # type: ignore
        HAS_SCIPY = True
    except Exception:  # pragma: no cover - optional
        HAS_SCIPY = False

    device = next(model.parameters()).device
    dt = float(env.dt)

    # Colors per-attractor (tab20 like in the notebook)
    import matplotlib.cm as cm
    true_points = env.points.cpu().numpy()

    # Estimate learned attractors numerically to build Voronoi regions.
    learned_points = _estimate_learned_attractors(
        model=model,
        grid_lim=grid_lim,
        num_samples=min(max(grid_n**2, 64), 100),
        num_steps=max(int(8.0 / dt), 75),
        tolerance=0.2,
        device=device,
    )
    print(
        f"[lyapunov] Learned attractor candidates: {learned_points.shape if learned_points.size else (0, 2)}",
        flush=True,
    )

    produced_files: Dict[str, str] = {}
    fig, axes = plt.subplots(1, 2, figsize=(20, 8))

    for ax, title, use_learned in (
        (axes[0], "True System", False),
        (axes[1], "Learned System", True),
    ):
        print(f"[lyapunov] Rendering '{title}' panel (use_learned={use_learned})", flush=True)
        display_points = (
            learned_points if use_learned and learned_points.size > 0 else true_points
        )
        num_points = max(len(display_points), 1)
        colors = cm.tab20(np.linspace(0, 1, num_points))

        # Optional Voronoi regions for both systems (true + learned estimate)
        if HAS_SCIPY and len(display_points) >= 3:
            vor = Voronoi(display_points)
            for i, point_idx in enumerate(vor.point_region):
                region = vor.regions[point_idx]
                if not region or -1 in region:
                    continue
                verts = np.array([vor.vertices[j] for j in region])
                if len(verts) > 0:
                    ax.fill(
                        verts[:, 0],
                        verts[:, 1],
                        color=colors[i % len(colors)],
                        alpha=0.2 if use_learned else 0.25,
                        zorder=1,
                    )
            for simplex in vor.ridge_vertices:
                simplex = np.asarray(simplex)
                if np.all(simplex >= 0):
                    ax.plot(
                        vor.vertices[simplex, 0],
                        vor.vertices[simplex, 1],
                        'k-',
                        linewidth=1.0,
                        alpha=0.7 if use_learned else 0.8,
                        zorder=2,
                    )

        # Grid and vector field (approximate using one-step delta / dt)
        xs = np.linspace(-grid_lim, grid_lim, grid_n)
        ys = np.linspace(-grid_lim, grid_lim, grid_n)
        X, Y = np.meshgrid(xs, ys)
        U = np.zeros_like(X)
        V = np.zeros_like(Y)

        for i in range(grid_n):
            for j in range(grid_n):
                state_np = np.array([X[i, j], Y[i, j]], dtype=np.float32)
                state_t = torch.from_numpy(state_np)
                if use_learned:
                    with torch.no_grad():
                        nx = model.step_env(state_t.to(device).unsqueeze(0)).squeeze(0).cpu()
                else:
                    nx = env.step(state_t)
                vel = (nx - state_t) / dt
                U[i, j], V[i, j] = float(vel[0].item()), float(vel[1].item())

        magnitudes = np.sqrt(U**2 + V**2)
        scale_den = np.where(magnitudes == 0, 1.0, magnitudes)
        U_n, V_n = U / scale_den, V / scale_den
        max_mag = float(magnitudes.max()) if magnitudes.size else 0.0
        linewidths = (
            0.75 + 2.25 * (magnitudes / (max_mag + 1e-6))
            if max_mag > 0
            else np.full_like(magnitudes, 0.75)
        )
        ax.quiver(
            X,
            Y,
            U_n,
            V_n,
            color='gray',
            alpha=0.65,
            scale=25,
            linewidths=linewidths.ravel(),
            zorder=3,
        )

        hist_suffix = "learned" if use_learned else "true"
        hist_path = path.parent / f"phase_portrait_vector_hist_{hist_suffix}.png"
        print(
            f"[lyapunov] Saving vector magnitude histogram ({hist_suffix}) to {hist_path}",
            flush=True,
        )
        _save_vector_magnitude_histogram(
            magnitudes,
            hist_path,
            title=f"{title} vector magnitudes",
        )
        produced_files[f"phase_portrait_vector_hist_{hist_suffix}"] = str(hist_path)

        marker_style = 's' if use_learned else 'o'
        for k, p in enumerate(display_points):
            ax.plot(
                p[0],
                p[1],
                marker_style,
                color=colors[k % len(colors)],
                markersize=10,
                markeredgecolor='black',
                markeredgewidth=2,
                zorder=6,
            )

        # Simulate trajectories from random initial conditions
        rng = np.random.default_rng(42)
        comparison_points = display_points if len(display_points) > 0 else true_points
        for _ in range(num_trajectories):
            x0 = rng.uniform(-2.5, 2.5, size=2).astype(np.float32)
            state = torch.from_numpy(x0)
            traj = [x0.copy()]
            steps = int(8.0 / dt)
            for _step in range(steps):
                if use_learned:
                    with torch.no_grad():
                        state = model.step_env(state.to(device).unsqueeze(0)).squeeze(0).cpu()
                else:
                    state = env.step(state)
                traj.append(state.numpy().copy())

            traj_arr = np.asarray(traj)
            final = traj_arr[-1]
            dists = np.linalg.norm(comparison_points - final, axis=1)
            idx = int(np.argmin(dists))
            color = colors[idx % len(colors)]
            ax.plot(traj_arr[:, 0], traj_arr[:, 1], color=color, lw=2.0, alpha=0.9, zorder=4)
            ax.plot(
                x0[0],
                x0[1],
                marker_style,
                color=color,
                markersize=6,
                alpha=0.9,
                markeredgecolor='white',
                markeredgewidth=1,
                zorder=5,
            )

        ax.set_xlim(-grid_lim, grid_lim)
        ax.set_ylim(-grid_lim, grid_lim)
        ax.set_xlabel('x1', fontsize=12)
        ax.set_ylabel('x2', fontsize=12)
        ax.set_title(
            title if not use_learned else f"{title} (Voronoi est.)",
            fontsize=14,
        )
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"[lyapunov] Phase portrait comparison saved to {path}", flush=True)

    produced_files["phase_portrait_comparison"] = str(path)
    return produced_files

# ---------------------------------------------------------------------------
# Evaluation driver
# ---------------------------------------------------------------------------


def _make_km_env_n_step(
    model: KoopmanMachine,
    x: torch.Tensor,
    length: int,
    reencode_at_every: int,
) -> torch.Tensor:
    """Torch analogue of notebooks/koopman_copy.py::make_km_env_n_step."""
    device = next(model.parameters()).device
    x = x.to(device)

    with torch.no_grad():
        if reencode_at_every == 1:
            traj = []
            state = x
            for _ in range(length):
                state = model.step_env(state)
                traj.append(state.detach().cpu())
            return torch.stack(traj, dim=0)
        elif reencode_at_every == 0:
            latents = []
            latent = model.encode(x)
            for _ in range(length):
                latent = model.step_latent(latent)
                latents.append(latent)

            latents_stack = torch.stack(latents, dim=0)
            return model.decode(latents_stack).detach().cpu()
        else:
            assert length % reencode_at_every == 0, (
                "length must be divisible by reencode_at_every when > 1"
            )
            state = x
            num_slices = length // reencode_at_every
            chunks: List[torch.Tensor] = []
            for _ in range(num_slices):
                latent = model.encode(state)
                chunk_states = []
                z = latent
                for _ in range(reencode_at_every):
                    z = model.step_latent(z)
                    decoded = model.decode(z)
                    chunk_states.append(decoded.detach().cpu())
                chunk = torch.stack(chunk_states, dim=0)
                chunks.append(chunk)
                state = chunk[-1].to(device)
            return torch.cat(chunks, dim=0)

    raise RuntimeError("Failed to generate Koopman rollout")


def _save_jax_style_phase_portraits(
    model: KoopmanMachine,
    base_env,
    cfg: Config,
    settings: "EvaluationSettings",
    path: Path,
) -> None:
    """Replicate notebooks/koopman_copy.py phase-portrait generation exactly."""
    if base_env.observation_size < 2:
        return

    _ensure_matplotlib()
    import matplotlib.pyplot as plt

    batch_size = settings.phase_portrait_batch_size
    length = settings.phase_portrait_length
    reencode_periods = settings.phase_portrait_reencode_periods

    vec_env = VectorWrapper(base_env, batch_size)
    rng = torch.Generator().manual_seed(cfg.SEED + settings.seed_offset + 999)
    init_states = vec_env.reset(rng)  # CPU tensor

    trajectories = {}
    for period in reencode_periods:
        traj = _make_km_env_n_step(model, init_states, length, period)
        trajectories[period] = traj  # [length, batch, obs_dim] on CPU

    num_modes = len(reencode_periods)
    fig, axes = plt.subplots(
        1, num_modes, figsize=(6 * num_modes, 5), squeeze=False
    )

    for ax, period in zip(axes[0], reencode_periods):
        traj = trajectories[period]
        ax.plot(traj[:, :, 0], traj[:, :, 1])
        if period == 0:
            title = "reencode [x]"
        elif period == 1:
            title = "reencode @ 1"
        else:
            title = f"reencode @ {period}"
        ax.set_title(title)
        ax.set_xlabel("x1")
        ax.set_ylabel("x2")
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle=":", alpha=0.4)

    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    plt.close(fig)


@dataclass
class EvaluationSettings:
    """Container for evaluation hyper-parameters."""

    systems: Sequence[str] = (
        # "pendulum",
        "duffing",
        # "lotka_volterra",
        # "lorenz63",
        # "parabolic",
        "lyapunov",
    )
    horizons: Sequence[int] = (100, 1000)
    periodic_reencode_periods: Sequence[int] = (10, 25, 50, 100)
    batch_size: int = 100
    phase_portrait_samples: int = 20
    phase_portrait_length: int = 200
    phase_portrait_reencode_periods: Sequence[int] = (0, 1, 10, 25, 50)
    phase_portrait_batch_size: int = 256
    seed_offset: int = 12345


def evaluate_model(
    model: KoopmanMachine,
    cfg: Config,
    device: torch.device | str = "cuda",
    settings: Optional[EvaluationSettings] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Dict]:
    """Evaluate a trained Koopman model using the standardized protocol.

    Args:
        model: Trained Koopman machine.
        cfg: Configuration used during training (provides baseline hyper-params).
        device: Device for model inference.
        settings: Optional evaluation settings. Defaults to the research spec.
        output_dir: Optional path to save metrics and plots. When provided, the
            function writes ``metrics.json``, phase portraits, and MSE curves.

    Returns:
        Nested dictionary with metrics for each system and rollout mode.
    """

    if settings is None:
        settings = EvaluationSettings()

    print(
        f"[evaluate_model] Starting evaluation for systems={tuple(settings.systems)} "
        f"with horizons={tuple(settings.horizons)}",
        flush=True,
    )

    model = model.to(device)
    model.eval()

    max_horizon = max(settings.horizons)
    results: Dict[str, Dict] = {}

    for system in settings.systems:
        print(f"[evaluate_model] -> System '{system}': preparing environment...", flush=True)
        eval_cfg = Config.from_dict(cfg.to_dict())
        eval_cfg.ENV.ENV_NAME = system

        base_env = make_env(eval_cfg)
        if base_env.observation_size != model.observation_size:
            # Skip incompatible systems to avoid runtime errors
            print(
                f"[evaluate_model] -> System '{system}': skipped because "
                f"observation size {base_env.observation_size} != model {model.observation_size}",
                flush=True,
            )
            continue

        vec_env = VectorWrapper(base_env, settings.batch_size)
        rng = torch.Generator().manual_seed(cfg.SEED + settings.seed_offset)
        init_states = vec_env.reset(rng)  # CPU tensor

        # Generate ground truth trajectories (time-major)
        print(
            f"[evaluate_model] -> System '{system}': generating ground-truth trajectory "
            f"(batch={settings.batch_size}, horizon={max_horizon})",
            flush=True,
        )
        true_future = generate_trajectory(vec_env.step, init_states, length=max_horizon)

        # Prepare initial states on device for model rollout
        init_states_device = init_states.to(device)

        predictions: Dict[str, torch.Tensor] = {}
        print(
            f"[evaluate_model] -> System '{system}': running rollout modes...",
            flush=True,
        )
        predictions["no_reencode"] = rollout_no_reencode(model, init_states_device, max_horizon)
        predictions["every_step"] = rollout_every_step_reencode(model, init_states_device, max_horizon)

        for period in settings.periodic_reencode_periods:
            mode_name = f"periodic_{period}"
            predictions[mode_name] = rollout_periodic_reencode(
                model,
                init_states_device,
                max_horizon,
                period=period,
            )

        mode_metrics: Dict[str, Dict] = {}
        periodic_summary: Dict[str, Dict[str, float]] = {str(h): {} for h in settings.horizons}
        per_step_errors: Dict[str, torch.Tensor] = {}

        # Convert ground truth to match predictions for metric computation
        true_future_cpu = true_future.float()

        print(
            f"[evaluate_model] -> System '{system}': computing metrics for {len(predictions)} modes...",
            flush=True,
        )
        for mode_name, pred in predictions.items():
            pred_cpu = pred.detach().cpu().float()

            per_step_error = torch.norm(pred_cpu - true_future_cpu, dim=-1).mean(dim=1)
            per_step_errors[mode_name] = per_step_error

            squared_diff = torch.sum((pred_cpu - true_future_cpu) ** 2, dim=-1)
            squared_diff = torch.where(torch.isfinite(squared_diff), squared_diff, torch.nan)

            horizons_metrics = {}
            for horizon in settings.horizons:
                if system == "parabolic" and horizon > 100:
                    # Skip 1000-step metric for parabolic attractor
                    continue

                mean, std, per_ic, num_valid = _compute_horizon_mse(squared_diff, horizon)
                horizons_metrics[str(horizon)] = {
                    "mean": mean,
                    "std": std,
                    "num_valid": num_valid,
                    "values": per_ic,
                }

                if mode_name.startswith("periodic_") and num_valid > 0:
                    periodic_summary[str(horizon)][mode_name] = mean

            mode_metrics[mode_name] = {
                "horizons": horizons_metrics,
                "mse_curve": _cumulative_mse_curve(squared_diff),
            }

        # Determine best periodic reencoding period per horizon
        best_periodic: Dict[str, Dict[str, float]] = {}
        for horizon in settings.horizons:
            horizon_key = str(horizon)
            if system == "parabolic" and horizon > 100:
                continue

            candidates = periodic_summary[horizon_key]
            if not candidates:
                continue

            best_mode = min(candidates.items(), key=lambda item: item[1])
            best_periodic[horizon_key] = {
                "mode": best_mode[0],
                "mean": best_mode[1],
            }

        # Save qualitative plots when requested
        files: Dict[str, str] = {}
        if output_dir is not None:
            system_dir = output_dir / system
            system_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[evaluate_model] -> System '{system}': saving plots to {system_dir}",
                flush=True,
            )

            # JAX-style phase portrait grid (matches notebooks/koopman_copy.py)
            portrait_path = system_dir / "phase_portrait_plot_eval.png"
            _save_jax_style_phase_portraits(
                model=model,
                base_env=base_env,
                cfg=cfg,
                settings=settings,
                path=portrait_path,
            )
            files["phase_portrait_plot_eval"] = str(portrait_path)

            curves = {
                mode: data["mse_curve"]
                for mode, data in mode_metrics.items()
            }
            curve_path = system_dir / "mse_vs_horizon.png"
            _save_mse_curve_plot(curves, curve_path, settings.horizons)
            files["mse_curve"] = str(curve_path)

            # Per-mode error curves (analogous to notebook plot_eval)
            for mode_name, errors in per_step_errors.items():
                error_path = system_dir / f"error_curve_{mode_name}.png"
                _save_error_curve_single_mode(
                    errors,
                    error_path,
                    title=f"Per-step error ({mode_name})",
                )
                files[f"error_curve_{mode_name}"] = str(error_path)

            combined_error_path = system_dir / "error_curve_combined.png"
            _save_error_curve_combined(
                per_step_errors,
                combined_error_path,
                highlight_steps=settings.horizons,
            )
            files["error_curve_combined"] = str(combined_error_path)

            # Additional notebook-style comparison for Lyapunov system
            if system == "lyapunov":
                try:
                    lyap_env = make_env(eval_cfg)
                    comp_path = system_dir / "phase_portrait_comparison.png"
                    print(
                        "[evaluate_model] -> System 'lyapunov': generating comparison + hist plots...",
                        flush=True,
                    )
                    lyap_files = _save_lyapunov_phase_portrait_comparison(
                        model,
                        lyap_env,
                        comp_path,
                    )
                    files.update(lyap_files)
                except Exception as e:  # pragma: no cover - visualization best-effort
                    # Don't fail evaluation if visualization fails
                    print(f"[warn] Lyapunov comparison plot failed: {e}")

        results[system] = {
            "modes": mode_metrics,
            "best_periodic": best_periodic,
            "files": files,
        }

    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)
        metrics_path = output_dir / "metrics.json"
        with metrics_path.open("w") as f:
            json.dump(results, f, indent=2)
        results["metrics_file"] = str(metrics_path)

    print("[evaluate_model] Finished evaluation for all requested systems.", flush=True)
    return results


__all__ = [
    "EvaluationSettings",
    "evaluate_model",
    "rollout_every_step_reencode",
    "rollout_no_reencode",
    "rollout_periodic_reencode",
]


