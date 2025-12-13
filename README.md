# Koopman-MPC Portfolio Rebalancing

PyTorch-based research codebase for **Koopman Model Predictive Control (MPC)** applied to dynamic portfolio rebalancing. The project learns Koopman operator representations of financial market dynamics and uses them for multi-step return forecasting and convex MPC-based portfolio optimization.

## Overview

This repository implements:

- **Koopman Autoencoders** for learning linear latent dynamics from nonlinear financial time series
- **Model Predictive Control (MPC)** using predicted returns for optimal portfolio rebalancing
- **Backtesting Framework** comparing Koopman-MPC against baselines (Buy & Hold, Markowitz)

### Koopman Autoencoder Variants

- **GenericKM**: Standard Koopman autoencoder with MLP encoder
- **SparseKM**: Koopman autoencoder with L1 sparsity regularization
- **LISTAKM**: Learned Iterative Soft-Thresholding Algorithm (LISTA) based sparse encoder

---

## Portfolio Rebalancing with Koopman-MPC

### The Idea

Financial markets are complex nonlinear dynamical systems. The Koopman operator provides a way to represent these nonlinear dynamics as **linear dynamics in a higher-dimensional latent space**:

```
z_{t+1} = K z_t    (linear in latent space)
x_t = ψ(z_t)       (nonlinear decoder back to returns)
```

This allows us to:
1. **Forecast returns** by unrolling the linear Koopman dynamics
2. **Solve convex MPC** since the dynamics are linear in the lifted space
3. **Optimize portfolios** with clear constraints (budget, no shorting, turnover limits)

### Quick Start: Train and Backtest

```bash
# 1. Train the Koopman model on finance data
uv run python train.py --config finance_sparse --env finance --num_steps 10000

# 2. Run backtesting with the trained model (after training completes)
# The backtest runs automatically at the end of training, or manually:
uv run python -c "
from config import get_config
from train import train
from backtest import run_backtest, BuyAndHoldStrategy, KoopmanMPCStrategy, BacktestConfig, calculate_metrics
from mpc import MPCConfig
from data_finance import create_finance_env
import torch

# Load trained model
cfg = get_config('finance_sparse')
checkpoint = torch.load('runs/kae_finance/<timestamp>/checkpoint.pt')
from model import make_model
env = create_finance_env(from_config=cfg)
model = make_model(cfg, env.observation_size)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Run Koopman-MPC backtest
mpc_cfg = MPCConfig(horizon=5, cost_coeff=0.001)
bt_cfg = BacktestConfig(initial_capital=10000, horizon=5)
strategy = KoopmanMPCStrategy(model, mpc_cfg)
results = run_backtest(strategy, env, bt_cfg)
print(calculate_metrics(results))
"
```

### Training the Koopman Model

The `finance_sparse` config is pre-configured for financial data:

```bash
# Basic training
uv run python train.py --config finance_sparse --env finance --num_steps 10000

# With custom parameters
uv run python train.py \
  --config finance_sparse \
  --env finance \
  --num_steps 20000 \
  --batch_size 64 \
  --target_size 128 \
  --sparsity_coeff 0.01 \
  --device cuda
```

**Key training features:**
- **Discrete Koopman dynamics**: Uses `z_{t+1} = K z_t` (not ODE integration) since financial data is discrete daily
- **Sequence training**: Trains on sequences of length 10 for multi-step stability
- **Time-delay embedding**: Each observation `Y_t = [y_t, y_{t-1}, ..., y_{t-d+1}]` captures temporal context

### Understanding the Data Pipeline

```
Prices p_t → Log-returns y_t = log(p_t/p_{t-1}) → Standardize → Time-delay embed Y_t → Koopman AE
```

The `FinanceEnv` handles:
- Downloading stock data via Yahoo Finance API
- Computing log-returns and standardizing (train stats only)
- Creating time-delay embeddings
- Chronological train/val/test splits (no data leakage)

### MPC Formulation

At each rebalancing step, the MPC solves:

```
maximize  Σ_{k=1}^H [ log(w_k^T exp(ŷ_k)) - λ ||w_k - w_{k-1}||_1 ]
subject to:
    Σ w_k = 1           (budget constraint)
    w_k ≥ 0             (no shorting)
    ||w_k - w_{k-1}||_1 ≤ τ  (turnover limit)
```

Where `ŷ_k` are the Koopman-predicted log-returns.

### Backtesting Strategies

```python
from backtest import run_backtest, BuyAndHoldStrategy, KoopmanMPCStrategy
from mpc import MPCConfig

# Buy & Hold (equal weight)
bh_strategy = BuyAndHoldStrategy()
bh_results = run_backtest(bh_strategy, env, backtest_config)

# Koopman-MPC
mpc_config = MPCConfig(
    horizon=5,           # 5-day prediction horizon
    cost_coeff=0.001,    # 10bps transaction cost
    max_turnover=0.2,    # Max 20% turnover per step
    allow_short=False,
)
koopman_strategy = KoopmanMPCStrategy(model, mpc_config)
koopman_results = run_backtest(koopman_strategy, env, backtest_config)

# Compare
print("Koopman-MPC:", calculate_metrics(koopman_results))
print("Buy & Hold:", calculate_metrics(bh_results))
```

### Evaluation Metrics

The backtester computes:
- **Sharpe Ratio**: Risk-adjusted returns (annualized)
- **Max Drawdown**: Largest peak-to-trough decline
- **Average Turnover**: Mean daily portfolio turnover
- **Total Return**: Overall portfolio growth

---

## Quick Start (Dynamical Systems)

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
koopman-mpc-portfolio-rebalancing/
├── config.py              # Configuration system with presets
├── data.py                # Dynamical systems environments
├── data_finance.py        # Finance data pipeline (Yahoo Finance, embeddings)
├── model.py               # Koopman autoencoder models
├── train.py               # Training script (CLI + API)
├── mpc.py                 # Model Predictive Control solvers
├── backtest.py            # Backtesting engine and strategies
├── baselines.py           # Baseline strategies (Markowitz, DMD)
├── evaluation.py          # Model evaluation
├── plot_training_metrics.py # Visualization utilities
├── tests/                 # Unit tests
├── notebooks/             # Research notebooks
└── runs/                  # Training outputs and checkpoints
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

### `finance_sparse` - Finance Portfolio Rebalancing
```bash
python train.py --config finance_sparse --env finance --num_steps 10000
```
- **Model**: GenericKM with sparse encoding
- **Target size**: 128 (latent dimension)
- **Encoder**: [256, 128] MLP with ReLU + bias
- **Decoder**: [128] MLP (linear output)
- **Training**: Sequence-based (length 10) with discrete Koopman dynamics
- **Loss weights**: Residual (1.0), Reconstruction (0.5), Prediction (1.0), Sparsity (0.01)
- **Data**: 20 liquid US stocks, 2012-2024, with time-delay embedding (d=20)

## Environments

| Environment | Dimension | Description |
|------------|-----------|-------------|
| `finance` | 400D* | Financial log-returns with time-delay embedding |
| `duffing` | 2D | Duffing oscillator with two stable centers |
| `pendulum` | 2D | Simple pendulum |
| `lotka_volterra` | 2D | Predator-prey dynamics |
| `lorenz63` | 3D | Chaotic Lorenz attractor |
| `parabolic` | 2D | Parabolic attractor (analytical Koopman) |
| `lyapunov` | 2D | Multi-attractor system with Lyapunov dynamics |

*Finance dimension = n_assets × embedding_dim (default: 20 assets × 20 lags = 400)

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

