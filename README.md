# Koopman-MPC Portfolio Rebalancing

PyTorch-based research codebase for **Koopman Model Predictive Control (MPC)** applied to dynamic portfolio rebalancing. The project learns Koopman operator representations of financial market dynamics and uses them for multi-step return forecasting and convex MPC-based portfolio optimization.

## Overview

This repository implements:

- **Koopman Autoencoders** for learning linear latent dynamics from nonlinear financial time series
- **Model Predictive Control (MPC)** using predicted returns for optimal portfolio rebalancing
- **Backtesting Framework** comparing Koopman-MPC against baselines (Buy & Hold, Markowitz, linear MPC baselines)

---

## Portfolio Rebalancing with Koopman-MPC

### Quick Start: Train and Backtest

```bash
# 1. Train the Koopman model on finance data
uv run python train.py --config finance_sparse --env finance --num_steps 1000 --sequence_length 10

# 2. Run the experiment to get baselines, backtest results, and portfolio value plots
uv run python run_experiment.py
# if you want to specify a path you can:
uv run python run_experiment.py --path runs/kae_finance/YOUR_TIMESTAMP_FOLDER
```

### Training the Koopman Model

The `finance_sparse` config is pre-configured for financial data:

```bash
# Basic training
uv run python train.py --config finance_sparse --env finance --num_steps 100

# With custom parameters (see template.sbatch)
NUM_STEPS=5000
SEQ_LENGTH=3
EMBEDDING_DIM=6 # set this to 2x the sequence length
TARGET_SIZE=1024
ENCODER_LAYERS="1024,1024"
SPARSITY_COEFF=0.3
RES_COEFF=1.0
RECONST_COEFF=0.1
PRED_COEFF=0.5
LEARNING_RATE=1e-4
LOG_DIR="./runs/kae_finance_weekly_seq${SEQ_LENGTH}"

uv run python train.py \
    --config finance_sparse \
    --env finance \
    --num_steps $NUM_STEPS \
    --sequence_length $SEQ_LENGTH \
    --embedding_dim $EMBEDDING_DIM \
    --resample_weekly \
    --target_size $TARGET_SIZE \
    --encoder_layers $ENCODER_LAYERS \
    --sparsity_coeff $SPARSITY_COEFF \
    --res_coeff $RES_COEFF \
    --reconst_coeff $RECONST_COEFF \
    --pred_coeff $PRED_COEFF \
    --lr $LEARNING_RATE \
    --log_dir $LOG_DIR
```

### Understanding the Data Pipeline

```
Prices p_t → Log-returns x_t = log(p_t/p_{t-1}) → Standardize → Time-delay embed X_t → Koopman AE
```

The `FinanceEnv` handles:
- Downloading stock data via Yahoo Finance API
- Computing log-returns and standardizing (train stats only)
- Creating time-delay embeddings
- Chronological train/val/test splits (no data leakage)

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
└── runs/                  # Training outputs and checkpoints
```

*Finance dimension = n_assets × embedding_dim (default: 20 assets × 6 lags = 120)

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

## Unit tests

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