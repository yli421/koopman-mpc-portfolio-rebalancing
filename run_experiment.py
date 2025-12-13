"""
Run the full Koopman-MPC Finance Experiment with Baselines.

1. Loads the best trained model.
2. Runs Backtesting for:
   - Buy & Hold (Baseline 1)
   - Markowitz Mean-Variance (Baseline 2)
   - DMD-MPC (Linear Baseline 3)
   - Koopman-MPC (Our Method)
3. Generates Comparison Plots and Metrics Table.
"""

import torch
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import argparse
from datetime import datetime

from config import Config
from model import make_model
from data_finance import create_finance_env
from backtest import (
    run_backtest, 
    BacktestConfig, 
    MPCConfig,
    BuyAndHoldStrategy, 
    KoopmanMPCStrategy,
    calculate_metrics
)
from baselines import MarkowitzStrategy, DMDStrategy

def main():
    # 1. Load Model
    parser = argparse.ArgumentParser(description='Run experiment evaluation')
    parser.add_argument('--path', type=str, help='Path to experiment run directory', default=None)
    args = parser.parse_args()

    if args.path:
        run_dir = Path(args.path)
    else:
        # Find latest run automatically
        search_dirs = [Path("runs/kae_finance"), Path("runs/kae")]
        latest_run = None
        latest_time = None
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for d in search_dir.iterdir():
                if d.is_dir() and (d / "checkpoint.pt").exists():
                    try:
                        # Parse timestamp from directory name
                        run_time = datetime.strptime(d.name, "%Y%m%d-%H%M%S")
                        if latest_time is None or run_time > latest_time:
                            latest_time = run_time
                            latest_run = d
                    except ValueError:
                        continue
        
        if latest_run is None:
            raise ValueError("Could not find any valid run directories in runs/kae_finance or runs/kae")
            
        run_dir = latest_run
        print(f"Automatically selected latest run: {run_dir}")

    checkpoint_path = run_dir / "checkpoint.pt"
    
    print(f"Loading model from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    cfg = Config.from_dict(checkpoint['config'])
    
    # Create Environment
    env = create_finance_env(from_config=cfg)
    
    # Create Model
    model = make_model(cfg, env.observation_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # 2. Setup Backtest
    bt_config = BacktestConfig(
        initial_capital=10000.0,
        horizon=5,
        rebalance_freq=1,
        cost_coeff=0.001
    )
    
    mpc_config = MPCConfig(
        horizon=5,
        gamma=0.0, # Log Utility
        cost_coeff=0.001,
        max_turnover=0.5
    )
    
    # 3. Run Strategies
    results = {}
    metrics = {}
    
    # --- Buy & Hold ---
    print("\n[1/4] Running Buy & Hold Strategy...")
    bh_strat = BuyAndHoldStrategy()
    results['Buy & Hold'] = run_backtest(bh_strat, env, bt_config)
    metrics['Buy & Hold'] = calculate_metrics(results['Buy & Hold'])
    
    # --- Markowitz ---
    print("\n[2/4] Running Markowitz Strategy...")
    # Gamma=1.0 is standard risk aversion. 
    # Adjust cost_coeff to be consistent or slightly higher if turnover is crazy.
    mark_strat = MarkowitzStrategy(risk_aversion=1.0, cost_coeff=0.001)
    results['Markowitz'] = run_backtest(mark_strat, env, bt_config)
    metrics['Markowitz'] = calculate_metrics(results['Markowitz'])
    
    # --- DMD (Linear Koopman) ---
    print("\n[3/4] Running DMD Strategy...")
    # Fit DMD on training data
    train_data = env.train_dataset.data
    dmd_strat = DMDStrategy(train_data, mpc_config)
    results['DMD-MPC'] = run_backtest(dmd_strat, env, bt_config)
    metrics['DMD-MPC'] = calculate_metrics(results['DMD-MPC'])
    
    # --- Koopman-MPC (Deep) ---
    print("\n[4/4] Running Koopman-MPC Strategy...")
    kmpc_strat = KoopmanMPCStrategy(model, mpc_config)
    results['Koopman-MPC'] = run_backtest(kmpc_strat, env, bt_config)
    metrics['Koopman-MPC'] = calculate_metrics(results['Koopman-MPC'])
    
    # 4. Results
    print("\n" + "="*50)
    print("RESULTS SUMMARY")
    print("="*50)
    
    metrics_df = pd.DataFrame(metrics).T
    print(metrics_df)
    
    # Save Metrics
    metrics_df.to_csv(run_dir / "full_comparison_metrics.csv")
    
    # 5. Plot Equity Curves
    plt.figure(figsize=(12, 7))
    
    colors = {
        'Buy & Hold': 'gray',
        'Markowitz': 'blue',
        'DMD-MPC': 'green',
        'Koopman-MPC': 'red'
    }
    styles = {
        'Buy & Hold': '--',
        'Markowitz': '-',
        'DMD-MPC': '-',
        'Koopman-MPC': '-'
    }
    
    # Use dates from one of the results (they should be identical)
    dates = pd.to_datetime(results['Buy & Hold']['date'])
    
    for name, df in results.items():
        plt.plot(dates, df['portfolio_value'], 
                 label=f"{name} (Sharpe: {metrics[name]['Sharpe Ratio']:.2f})",
                 color=colors.get(name), linestyle=styles.get(name))
    
    plt.title("Portfolio Strategy Comparison (2021-2024)")
    plt.xlabel("Date")
    plt.ylabel("Portfolio Value ($)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = run_dir / "equity_curve_comparison.png"
    plt.savefig(plot_path)
    print(f"\nComparison plot saved to {plot_path}")

if __name__ == "__main__":
    main()
