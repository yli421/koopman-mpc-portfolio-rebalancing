"""Evaluation utilities for Koopman Autoencoder models.

This module provides rollout strategies for Koopman models, supporting
latent-only evolution, state-space evolution, and periodic re-encoding.
"""

from __future__ import annotations

import torch
from typing import List, Optional

from model import KoopmanMachine


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


__all__ = [
    "rollout_every_step_reencode",
    "rollout_no_reencode",
    "rollout_periodic_reencode",
]
