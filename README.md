# Sparse Koopman Autoencoder (SKAE)

PyTorch-based research codebase for learning Koopman operator representations of nonlinear dynamical systems using autoencoders with sparsity constraints.

## Overview

This repository implements several variants of Koopman autoencoders:

- **GenericKM**: Standard Koopman autoencoder with MLP encoder
- **SparseKM**: Koopman autoencoder with L1 sparsity regularization
- **LISTAKM**: Learned Iterative Soft-Thresholding Algorithm (LISTA) based sparse encoder

## Quick Start

### Installation

This project uses [`uv`](https://github.com/astral-sh/uv) for fast, reliable dependency management.

Installing uv on MacOS/Linux:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

To install uv for Windows, open PowerShell and run:
```bash
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex
```
The -ExecutionPolicy ByPass flag allows running the installation script from the internet.

**Install the project and dependencies**:
```bash
# Clone the repository (if you haven't already)
git clone <repository-url>
cd skae

# Install from lock file (reproducible, recommended)
uv sync

# Alternative: Install without lock file
uv pip install -e .
```

### Train a Model

```bash
# Train with defaults on the Duffing Oscillator
uv run python train.py --config generic_sparse --env duffing --pairwise --num_steps 20000

# Sweep over sparsity coefficient
uv run python train.py --config generic_sparse --env pendulum --sparsity_coeff 0.001
uv run python train.py --config generic_sparse --env pendulum --sparsity_coeff 0.01
uv run python train.py --config generic_sparse --env pendulum --sparsity_coeff 0.1

# Custom learning rate and latent dimension
uv run python train.py \
  --config generic_sparse \
  --env lyapunov \
  --num_steps 20000 \
  --batch_size 256 \
  --target_size 64 \
  --reconst_coeff 0.02 \
  --pred_coeff 1.0 \
  --sparsity_coeff 0.001 \
  --pairwise \
  --seed 0 \
  --device cuda

uv run python train.py \
  --config lista \
  --env lyapunov \
  --num_steps 20000 \
  --batch_size 256 \
  --target_size 64 \
  --reconst_coeff 0.02 \
  --pred_coeff 1.0 \
  --pairwise \
  --seed 0 \
  --device cuda


# Evaluation from checkpoints
# Required: specify system
python evaluate_checkpoints.py --run_dir runs/kae/<timestamp> --system duffing

# With other options
python evaluate_checkpoints.py --run_dir runs/kae/<timestamp> --system pendulum --device cpu
python evaluate_checkpoints.py --run_dir runs/kae/<timestamp> --system lorenz63 --checkpoints checkpoint.pt
```

## Repository Structure

```
skae/
├── config.py              # Configuration system with presets
├── data.py                # Dynamical systems environments
├── model.py               # Koopman autoencoder models
├── train.py               # Training script (CLI + API)
├── evaluation.py          # Model evaluation
├── plot_metrics.py        # Visualization utilities
├── tests/                 # Unit tests
├── notebooks/             # Research notebooks
```

## Available Configurations

### `generic` - Standard Koopman Autoencoder
```bash
python train.py --config generic --env duffing
```
- **Model**: GenericKM
- **Target size**: 64
- **Encoder**: [64, 64] MLP
- **Decoder**: Linear
- **Loss weights**: Residual (1.0), Reconstruction (0.02)

### `generic_sparse` - Sparse Koopman with L1 regularization
```bash
python train.py --config generic_sparse --env duffing --sparsity_coeff 0.01
```
- **Model**: GenericKM
- **Target size**: 64
- **Encoder**: [64, 64] MLP with ReLU + bias
- **Decoder**: Linear
- **Loss weights**: Residual (1.0), Reconstruction (0.5), Sparsity (0.01)

### `generic_prediction` - Prediction-focused
```bash
python train.py --config generic_prediction --env duffing
```
- **Loss weights**: Prediction (1.0), others disabled

### `lista` - LISTA Sparse Encoder
```bash
python train.py --config lista --env lotka_volterra --target_size 2048
```
- **Model**: LISTAKM
- **Target size**: 2048 (overcomplete)
- **Encoder**: LISTA with 10 iterations
- **Decoder**: Normalized dictionary
- **Loss weights**: Residual (1.0), Reconstruction (1.0), Sparsity (1.0)

### `lista_nonlinear` - LISTA with MLP
```bash
python train.py --config lista_nonlinear --env lorenz63
```
- **Model**: LISTAKM with nonlinear pre-activation
- **Encoder**: [64, 64, 64] MLP → LISTA

## Environments

| Environment | Dimension | Description |
|------------|-----------|-------------|
| `duffing` | 2D | Duffing oscillator with two stable centers |
| `pendulum` | 2D | Simple pendulum |
| `lotka_volterra` | 2D | Predator-prey dynamics |
| `lorenz63` | 3D | Chaotic Lorenz attractor |
| `parabolic` | 2D | Parabolic attractor (analytical Koopman) |
| `lyapunov` | 2D | Multi-attractor system with Lyapunov dynamics |

## Training Output

Each training run creates a timestamped directory:

```
runs/kae/20251106-223912/
├── config.json              # Full configuration (reproducibility)
├── checkpoint.pt            # Best model (lowest validation error)
├── last.pt                  # Latest checkpoint
├── metrics_history.jsonl    # Time series of all metrics
├── metrics_summary.json     # Summary statistics
└── final_metrics.json       # Final step metrics
```

## Model Evaluation

The evaluation module (`evaluation.py`) provides comprehensive evaluation of trained Koopman models using multiple rollout strategies and horizon-wise metrics.

### Automatic Evaluation

Evaluation runs automatically at the end of training and saves results to `runs/kae/<timestamp>/evaluation/`. The evaluation protocol tests models on multiple dynamical systems, computes horizon-wise mean-squared error metrics, and generates qualitative plots.

### Standalone Evaluation

You can also evaluate trained checkpoints independently using `evaluate_checkpoints.py`:

```bash
# Evaluate a checkpoint on a specific system
uv run python evaluate_checkpoints.py --run_dir runs/kae/<timestamp> --system duffing

# Evaluate multiple checkpoints
uv run python evaluate_checkpoints.py \
  --run_dir runs/kae/<timestamp> \
  --system pendulum \
  --checkpoints checkpoint.pt last.pt \
  --device cuda

# Evaluate on CPU
uv run python evaluate_checkpoints.py --run_dir runs/kae/<timestamp> --system lyapunov --device cpu
```

### Rollout Strategies

The evaluation protocol tests three rollout modes:

1. **No reencoding** (`no_reencode`): Evolves entirely in latent space using `step_latent()` without reencoding
2. **Every-step reencoding** (`every_step`): Reencodes at each step using `step_env()` (state-space evolution)
3. **Periodic reencoding** (`periodic_k`): Reencodes every k steps (default periods: 10, 25, 50, 100)

For periodic reencoding, the evaluation automatically selects the best period per horizon based on MSE.

### Evaluation Metrics

For each system and rollout mode, the evaluation computes:

- **Horizon-wise MSE**: Mean ± std MSE aggregated across initial conditions for horizons (default: 100, 1000 steps)
- **Cumulative MSE curve**: Time-averaged MSE vs. prediction horizon
- **Per-step L2 error**: Mean L2 error at each prediction step
- **Best periodic reencoding**: Automatically identifies optimal reencoding period per horizon

Metrics are computed over a batch of unseen initial conditions (default: 100 samples) and handle exploding rollouts gracefully by marking them as NaN.

### Evaluation Output

The evaluation generates the following outputs in `runs/kae/<timestamp>/evaluation/<system>/`:

**Metrics:**
- `metrics.json`: Structured JSON with all metrics organized by system, mode, and horizon

**Plots:**
- `phase_portrait_plot_eval.png`: Grid of phase portraits for different reencoding periods
- `mse_vs_horizon.png`: Cumulative MSE curves for all rollout modes
- `error_curve_<mode>.png`: Per-step error curves for each mode
- `error_curve_combined.png`: Combined per-step error curves for all modes

**Special plots for Lyapunov system:**
- `phase_portrait_comparison.png`: Side-by-side comparison of true vs. learned system with Voronoi regions, vector fields, and trajectories
- `phase_portrait_vector_hist_true.png`: Histogram of vector field magnitudes (true system)
- `phase_portrait_vector_hist_learned.png`: Histogram of vector field magnitudes (learned system)

The `metrics.json` structure:
```json
{
  "<system>": {
    "modes": {
      "no_reencode": {
        "horizons": {
          "100": {"mean": <float>, "std": <float>, "num_valid": <int>, "values": [<float>]},
          "1000": {...}
        },
        "mse_curve": [<float>]
      },
      "every_step": {...},
      "periodic_10": {...},
      ...
    },
    "best_periodic": {
      "100": {"mode": "periodic_25", "mean": <float>},
      "1000": {...}
    },
    "files": {
      "phase_portrait_plot_eval": "<path>",
      "mse_curve": "<path>",
      ...
    }
  }
}
```


## Testing

```bash
# Run all tests
pytest

# Run specific test suite
pytest tests/test_train.py -v

# Run with coverage
pytest --cov=. --cov-report=html
```

## License

See `LICENSE` file for details.

