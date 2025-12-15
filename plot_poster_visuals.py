
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, List, Tuple

from config import Config
from data import make_env
from model import make_model
from data_finance import create_finance_env, FinanceEnv
from evaluation import rollout_no_reencode, rollout_every_step_reencode, rollout_periodic_reencode
from backtest import run_backtest, BacktestConfig, MPCConfig, BuyAndHoldStrategy, KoopmanMPCStrategy, calculate_metrics
from baselines import MarkowitzStrategy, DMDStrategy, VARStrategy

def load_checkpoint(run_dir: Path, checkpoint_name: str = "last.pt") -> Tuple[torch.nn.Module, Config, Dict]:
    """Load model and config from checkpoint."""
    checkpoint_path = run_dir / checkpoint_name
    
    # Strictly prefer the requested checkpoint name first
    if not checkpoint_path.exists():
        print(f"Warning: {checkpoint_path} not found.")
        # Fallback options
        if (run_dir / "best.pt").exists():
            checkpoint_path = run_dir / "best.pt"
            print(f"Falling back to {checkpoint_path}")
        elif (run_dir / "checkpoint.pt").exists():
            checkpoint_path = run_dir / "checkpoint.pt"
            print(f"Falling back to {checkpoint_path}")
        else:
            raise FileNotFoundError(f"No checkpoint found in {run_dir} (looked for {checkpoint_name}, best.pt, checkpoint.pt)")
    
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    
    # Load config
    if (run_dir / "config.json").exists():
        cfg = Config.from_json(str(run_dir / "config.json"))
    elif 'config' in checkpoint:
        cfg = Config.from_dict(checkpoint['config'])
    else:
        raise ValueError("Could not find configuration")
        
    return checkpoint, cfg

def plot_forecast_trajectories(
    model: torch.nn.Module,
    env: FinanceEnv,
    device: str,
    output_dir: Path,
    horizon: int = 50,
    n_assets_to_plot: int = 5
):
    """Plot actual vs predicted price trajectories (1-step lookahead)."""
    print(f"Generating forecast trajectories for {n_assets_to_plot} assets...")
    model.eval()
    model = model.to(device)
    
    # Get a long sequence from test set
    test_data = env.test_dataset.data  # [T, obs_size]
    test_dates = env.test_dataset.dates
    
    if len(test_data) < horizon + 10:
        print("Test data too short for forecast plot")
        return

    # Start from a point where we have interesting movement
    start_idx = len(test_data) // 4
    
    # Use ground truth history for 1-step lookahead predictions (Teacher Forcing)
    # Input: x_t, x_{t+1}, ... x_{t+H-1}
    # Output: x_{t+1}, x_{t+2}, ... x_{t+H} (predicted)
    
    inputs = test_data[start_idx : start_idx + horizon].to(device) # [horizon, obs_size]
    
    with torch.no_grad():
        # Predict next state given current real state
        preds_flat = model.step_env(inputs) # [horizon, obs_size]
        predictions = preds_flat.unsqueeze(1) # [horizon, 1, obs_size]
        
    # Get ground truth
    ground_truth = test_data[start_idx+1 : start_idx+1+horizon].unsqueeze(1)
    
    # Extract returns
    pred_returns = env.extract_current_returns(predictions).squeeze(1).cpu() # [horizon, n_assets]
    true_returns = env.extract_current_returns(ground_truth).squeeze(1).cpu()
    
    # Destandardize
    pred_returns = env.destandardize_returns(pred_returns).numpy()
    true_returns = env.destandardize_returns(true_returns).numpy()
    
    # Convert to Price Index (Cumulative Return)
    pred_price = np.exp(np.cumsum(pred_returns, axis=0))
    true_price = np.exp(np.cumsum(true_returns, axis=0))
    
    # Add initial point (1.0)
    pred_price = np.vstack([np.ones((1, pred_price.shape[1])), pred_price])
    true_price = np.vstack([np.ones((1, true_price.shape[1])), true_price])
    
    # Dates
    plot_dates = test_dates[start_idx : start_idx + horizon + 1]
    
    # Plot
    tickers = env.metadata.get('tickers', [f'Asset {i}' for i in range(env.n_assets)])
    print(tickers)
    # Select specific assets to plot
    # We want ABBV (Healthcare) and others from different sectors
    # Common liquid ones in our list: AAPL (Tech), JPM (Finance), XOM (Energy)
    target_tickers = ['AAPL', 'BAC', 'GOOGL', 'GS', 'KO']
    indices_to_plot = []
    
    for t in target_tickers:
        if t in tickers:
            print(t)
            indices_to_plot.append(tickers.index(t))
    print(
        "indices to plot", indices_to_plot
    )
    # If we didn't find enough specific ones, fill with others
    if len(indices_to_plot) < n_assets_to_plot:
        remaining = n_assets_to_plot - len(indices_to_plot)
        for i in range(env.n_assets):
            if i not in indices_to_plot:
                indices_to_plot.append(i)
                if len(indices_to_plot) >= n_assets_to_plot:
                    break
    
    # indices_to_plot = indices_to_plot[:n_assets_to_plot]
    
    # Use all assets or limit to requested number
    n_plot = len(indices_to_plot)
    
    # Create subplots - careful with figure size for many assets
    fig, axes = plt.subplots(n_plot, 1, figsize=(12, 3 * n_plot), sharex=True)
    if n_plot == 1:
        axes = [axes]
        
    for i in range(n_plot):
        ax = axes[i]
        idx = indices_to_plot[i]
        ax.plot(plot_dates, true_price[:, idx], label='Actual Price', color='black', linewidth=1.5, alpha=0.7)
        ax.plot(plot_dates, pred_price[:, idx], label='Koopman Forecast', color='#e74c3c', linewidth=2, linestyle='--')
        
        ax.set_title(f"{tickers[idx]} - {horizon}-Step Forecast Sequence", fontsize=12)
        ax.set_ylabel("Price Index ($)")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(loc='upper left')
            
    plt.xlabel("Date")
    fig.autofmt_xdate()
    plt.tight_layout()
    
    save_path = output_dir / "poster_forecast_trajectories.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close(fig)


def plot_strategy_backtest(
    model: torch.nn.Module,
    env: FinanceEnv,
    device: str,
    output_dir: Path
):
    """Backtest a simple directional strategy (Re-encode every step)."""
    print("Running strategy backtest...")
    model.eval()
    model = model.to(device)
    
    test_data = env.test_dataset.data.to(device)
    test_dates = env.test_dataset.dates
    n_steps = len(test_data) - 1
    
    # x0: [n_steps, obs_size]
    x0 = test_data[:-1]
    
    batch_size = 256
    all_preds = []
    
    with torch.no_grad():
        for i in range(0, len(x0), batch_size):
            batch = x0[i:i+batch_size]
            # Predict next step
            pred_batch = model.step_env(batch)
            all_preds.append(pred_batch.cpu())
            
    predictions = torch.cat(all_preds, dim=0) # [n_steps, obs_size]
    
    # Extract returns
    pred_returns = env.extract_current_returns(predictions) # [n_steps, n_assets] (Standardized)
    
    # Get actual next-step returns
    true_next_step = test_data[1:].cpu()
    true_returns = env.extract_current_returns(true_next_step) # [n_steps, n_assets] (Standardized)
    
    # De-standardize for calculation
    pred_ret_real = env.destandardize_returns(pred_returns).numpy()
    true_ret_real = env.destandardize_returns(true_returns).numpy()
    
    n_assets = env.n_assets
    
    # Benchmark Returns (Equal Weight)
    bench_weights = np.ones_like(true_ret_real) / n_assets
    bench_portfolio_ret = np.sum(bench_weights * true_ret_real, axis=1)
    
    # Strategy Returns
    # Simple logic: if pred > 0, buy. Normalize weights to sum to 1?
    # Or just invest in positive assets equally?
    # Let's say we have capital 1. We allocate 1/N to each asset if pred > 0. 
    # If pred <= 0, we keep that portion in cash (ret=0).
    
    signal = (pred_ret_real > 0).astype(float)
    strat_weights = signal / n_assets 
    strat_portfolio_ret = np.sum(strat_weights * true_ret_real, axis=1)
    
    # Cumulative Returns
    bench_cum = np.cumsum(bench_portfolio_ret)
    strat_cum = np.cumsum(strat_portfolio_ret)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    dates = test_dates[1:]
    
    ax.plot(dates, np.exp(bench_cum), label='Benchmark (Buy & Hold)', color='gray', linewidth=1.5)
    ax.plot(dates, np.exp(strat_cum), label='Koopman Directional Strategy', color='#2ecc71', linewidth=2)
    
    ax.set_title("Strategy Performance: Koopman 1-Step Forecast", fontsize=14)
    ax.set_ylabel("Portfolio Value ($)")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Add Sharpe Ratio annotation
    def sharpe(rets):
        return np.mean(rets) / (np.std(rets) + 1e-8) * np.sqrt(52 if env.metadata.get('resample_weekly') else 252)
        
    s_bench = sharpe(bench_portfolio_ret)
    s_strat = sharpe(strat_portfolio_ret)
    
    textstr = f'Sharpe Ratios:\nBenchmark: {s_bench:.2f}\nKoopman: {s_strat:.2f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
            
    fig.autofmt_xdate()
    plt.tight_layout()
    
    save_path = output_dir / "poster_strategy_backtest.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close(fig)

def plot_mpc_receding_horizon(
    model: torch.nn.Module,
    env: FinanceEnv,
    device: str,
    output_dir: Path,
    horizon: int = 10
):
    """Plot receding horizon forecasts to illustrate MPC mechanism."""
    print("Generating MPC receding horizon plot...")
    model.eval()
    model = model.to(device)
    
    # Pick a segment
    test_data = env.test_dataset.data # [T, obs_size]
    test_dates = env.test_dataset.dates
    
    start_idx = len(test_data) // 2
    n_steps = 30
    
    if len(test_data) < start_idx + n_steps + horizon:
        print("Data too short for MPC plot")
        return
        
    segment_data = test_data[start_idx : start_idx + n_steps + horizon]
    segment_dates = test_dates[start_idx : start_idx + n_steps + horizon]
    
    # Extract actual prices for one asset
    asset_idx = 0
    tickers = env.metadata.get('tickers', [])
    asset_name = tickers[asset_idx] if tickers else "Asset 0"
    
    # Get actual price path
    true_returns = env.extract_current_returns(segment_data).cpu()
    true_returns = env.destandardize_returns(true_returns).numpy()
    true_price = np.exp(np.cumsum(true_returns, axis=0))
    # Normalize start to 1.0
    true_price = true_price / true_price[0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Plot actual path
    ax.plot(segment_dates[:n_steps+horizon], true_price[:, asset_idx], 
            label='Actual Price', color='black', linewidth=2, zorder=10)
            
    # Plot forecasts at intervals
    interval = 5
    colors = plt.cm.viridis(np.linspace(0, 1, n_steps // interval + 1))
    
    for i in range(0, n_steps, interval):
        # Current state
        x0 = segment_data[i:i+1].to(device)
        
        # Forecast
        with torch.no_grad():
            preds = rollout_no_reencode(model, x0, horizon)
            
        # Process forecast returns
        pred_rets = env.extract_current_returns(preds).squeeze(1).cpu()
        pred_rets = env.destandardize_returns(pred_rets).numpy()
        
        # Convert to price path starting from current actual price
        current_price = true_price[i, asset_idx]
        
        # Select specific asset returns
        asset_pred_rets = pred_rets[:, asset_idx]
        cum_ret = np.cumsum(asset_pred_rets)
        
        pred_price_path = current_price * np.exp(cum_ret)
        pred_price_path = np.insert(pred_price_path, 0, current_price) # prepend current
        
        # Dates for forecast
        fcast_dates = segment_dates[i : i + horizon + 1]
        
        ax.plot(fcast_dates, pred_price_path, 
                color=colors[i//interval], linestyle='--', linewidth=2, alpha=0.8,
                label=f'Forecast @ t={i}')
                
        # Add dot at start
        ax.scatter(segment_dates[i], current_price, color=colors[i//interval], s=50, zorder=11)
        
    ax.set_title(f"Receding Horizon Forecasts (MPC Concept) - {asset_name}", fontsize=14)
    ax.set_ylabel("Normalized Price")
    ax.set_xlabel("Date")
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    fig.autofmt_xdate()
    plt.tight_layout()
    
    save_path = output_dir / "poster_mpc_mechanism.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close(fig)

def plot_eigenvalues(
    model: torch.nn.Module,
    device: str,
    output_dir: Path
):
    """Plot eigenvalues of the learned Koopman operator."""
    print("Generating eigenvalue plot...")
    model.eval()
    model = model.to(device)
    
    with torch.no_grad():
        kmat = model.kmatrix()
        if device == 'cuda':
            kmat = kmat.cpu()
        eigvals = torch.linalg.eigvals(kmat)
        
    real = eigvals.real.numpy()
    imag = eigvals.imag.numpy()
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.5, label='Unit Circle')
    
    # Eigenvalues
    ax.scatter(real, imag, color='#8e44ad', alpha=0.6, s=50, label='Koopman Eigenvalues')
    
    # Highlight max magnitude
    max_mag = np.max(np.abs(eigvals.numpy()))
    ax.text(0.05, 0.95, f"Max Magnitude: {max_mag:.4f}", 
            transform=ax.transAxes, bbox=dict(facecolor='wheat', alpha=0.5))
    
    ax.set_title("Spectrum of Learned Koopman Operator", fontsize=14)
    ax.set_xlabel("Real Part")
    ax.set_ylabel("Imaginary Part")
    ax.grid(True, alpha=0.3)
    ax.axis('equal')
    ax.legend()
    
    save_path = output_dir / "poster_eigenvalues.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close(fig)

def plot_risk_return_frontier(
    model: torch.nn.Module,
    env: FinanceEnv,
    device: str,
    output_dir: Path,
    cfg: Config
):
    """Plot Risk-Return Frontier comparing strategies."""
    print("Generating Risk-Return Frontier plot...")
    
    # 1. Setup Configs
    bt_config = BacktestConfig(
        initial_capital=10000.0,
        horizon=5,
        rebalance_freq=1,
        cost_coeff=0.001
    )
    
    # MPC Config (Log Utility)
    mpc_config = MPCConfig(
        horizon=5,
        gamma=0.0,
        cost_coeff=0.001,
        max_turnover=0.2,
        risk_free_rate=0.0
    )
    
    # Determine frequency from Config, not env metadata
    is_weekly = cfg.ENV.FINANCE.RESAMPLE_WEEKLY
    freq_factor = 52 if is_weekly else 252
    print(f"  Detected frequency: {'Weekly' if is_weekly else 'Daily'} (Factor: {freq_factor})")
    
    strategies = {}
    
    # 2. Instantiate Strategies
    # Buy & Hold
    strategies['Buy & Hold'] = {
        'strategy': BuyAndHoldStrategy(),
        'color': 'gray',
        'marker': 's'
    }
    
    # Markowitz (Mean-Variance)
    strategies['Markowitz'] = {
        'strategy': MarkowitzStrategy(risk_aversion=1.0, cost_coeff=0.001),
        'color': 'blue',
        'marker': '^'
    }
    
    # DMD-MPC (Linear Baseline)
    # Fit DMD on training data
    train_data = env.train_dataset.data # [N, obs_size]
    strategies['DMD-MPC'] = {
        'strategy': DMDStrategy(train_data, mpc_config),
        'color': 'green',
        'marker': 'v'
    }

    # VAR-MPC (Baseline 4)
    lags = min(env.embedding_dim, 5)
    strategies['VAR-MPC'] = {
        'strategy': VARStrategy(train_data, mpc_config, n_assets=env.n_assets, lags=lags),
        'color': 'purple',
        'marker': 'D'
    }
    
    # Koopman-MPC (Our Method)
    model.eval()
    model = model.to(device)
    strategies['Koopman-MPC'] = {
        'strategy': KoopmanMPCStrategy(model, mpc_config, device=device),
        'color': 'red',
        'marker': '*'
    }
    
    # 3. Run Backtests & Collect Metrics
    results = []
    
    for name, info in strategies.items():
        print(f"  Running backtest for {name}...")
        df = run_backtest(info['strategy'], env, bt_config, verbose=False)
        
        # Calculate annualized metrics
        returns = df['return'].values
        ann_return = np.mean(returns) * freq_factor
        ann_volatility = np.std(returns) * np.sqrt(freq_factor)
        sharpe = ann_return / (ann_volatility + 1e-8)
        
        results.append({
            'name': name,
            'return': ann_return,
            'risk': ann_volatility,
            'sharpe': sharpe,
            'color': info['color'],
            'marker': info['marker']
        })
        print(f"    {name}: Return={ann_return:.4f}, Risk={ann_volatility:.4f}, Sharpe={sharpe:.4f}")
        
    # 4. Plot
    fig, ax = plt.subplots(figsize=(8, 6), dpi=300)
    
    for res in results:
        ax.scatter(res['risk'], res['return'], 
                   color=res['color'], marker=res['marker'], s=200, 
                   edgecolors='black', label=res['name'], zorder=10)
                   
    # Find Koopman result for annotation
    k_res = next(r for r in results if r['name'] == 'Koopman-MPC')
    
    # Add text annotation
    ax.annotate(f'Koopman-MPC\nSharpe: {k_res["sharpe"]:.2f}', 
                xy=(k_res['risk'], k_res['return']), 
                xytext=(k_res['risk']*1.1, k_res['return']),
                arrowprops=dict(facecolor='black', shrink=0.05),
                fontsize=10, fontweight='bold')

    ax.set_title("Strategy Risk-Return Comparison (top-left is better)", fontsize=16)
    ax.set_xlabel("Annualized Volatility (Risk)", fontsize=12)
    ax.set_ylabel("Annualized Return", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower right', frameon=True, framealpha=1, shadow=True)
    
    # Add a thin dotted line between Buy & Hold and Markowitz to show Pareto improvement
    bh_res = next(r for r in results if r['name'] == 'Buy & Hold')
    mk_res = next(r for r in results if r['name'] == 'Markowitz')
    var_res = next(r for r in results if r['name'] == 'VAR-MPC')
    
    # Naive Frontier (B&H -> Markowitz)
    # ax.plot([bh_res['risk'], mk_res['risk']], [bh_res['return'], mk_res['return']], 
    #         color='gray', linestyle=':', linewidth=1.0, alpha=0.6, label='Naive Frontier')

    # VAR Connections
    ax.plot([bh_res['risk'], var_res['risk']], [bh_res['return'], var_res['return']], 
            color='purple', linestyle='--', linewidth=1.0, alpha=0.5)
    ax.plot([mk_res['risk'], var_res['risk']], [mk_res['return'], var_res['return']], 
            color='purple', linestyle='--', linewidth=1.0, alpha=0.5)
            
    # Adjust limits to look nice
    risks = [r['risk'] for r in results]
    rets = [r['return'] for r in results]
    
    # Add margins
    ax.set_xlim(min(risks) * 0.8, max(risks) * 1.2)
    ax.set_ylim(min(min(rets), 0) * 1.2, max(rets) * 1.2)
    
    plt.tight_layout()
    save_path = output_dir / "poster_frontier.png"
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('run_dir', type=str, help='Run directory')
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load
    checkpoint, cfg = load_checkpoint(run_dir, checkpoint_name="last.pt")
    
    # Create Env
    env = create_finance_env(from_config=cfg)
    
    # Create Model
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Plots
    plot_forecast_trajectories(model, env, device, run_dir, horizon=50)
    plot_strategy_backtest(model, env, device, run_dir)
    plot_mpc_receding_horizon(model, env, device, run_dir)
    plot_eigenvalues(model, device, run_dir)
    plot_risk_return_frontier(model, env, device, run_dir, cfg)

if __name__ == '__main__':
    main()
