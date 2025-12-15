"""
Run the full Koopman-MPC Finance Experiment with Baselines.

1. Loads trained model(s) - both Best and Last checkpoints if available.
2. Runs Backtesting for:
   - Buy & Hold (Baseline 1)
   - Markowitz Mean-Variance (Baseline 2)
   - DMD-MPC (Linear Baseline 3)
   - Koopman-MPC (Our Method) - Best and/or Last
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
    # 1. Locate Run Directory
    parser = argparse.ArgumentParser(description='Run experiment evaluation')
    parser.add_argument('--path', type=str, help='Path to experiment run directory', default=None)
    parser.add_argument('--search_dir', type=str, help='Directory to search for latest run', default=None)
    args = parser.parse_args()

    if args.path:
        run_dir = Path(args.path)
    else:
        # Find latest run automatically
        if args.search_dir:
            search_dirs = [Path(args.search_dir)]
        else:
            search_dirs = [Path("runs/kae_finance"), Path("runs/kae")]
        latest_run = None
        latest_time = None
        
        for search_dir in search_dirs:
            if not search_dir.exists():
                continue
            for d in search_dir.iterdir():
                if d.is_dir() and ((d / "checkpoint.pt").exists() or (d / "last.pt").exists()):
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

    # Check which checkpoints exist
    checkpoints_to_eval = []
    if (run_dir / "checkpoint.pt").exists():
        checkpoints_to_eval.append(("Best", run_dir / "checkpoint.pt"))
    if (run_dir / "last.pt").exists():
        checkpoints_to_eval.append(("Last", run_dir / "last.pt"))
    
    if not checkpoints_to_eval:
        raise FileNotFoundError(f"No checkpoint.pt or last.pt found in {run_dir}")

    # Load Config and Environment from the first available checkpoint
    # (Config should be same for both)
    print(f"Loading config from {checkpoints_to_eval[0][1]}...")
    first_ckpt = torch.load(checkpoints_to_eval[0][1], map_location='cpu')
    cfg = Config.from_dict(first_ckpt['config'])
    
    # Create Environment
    env = create_finance_env(from_config=cfg)
    
    # 2. Setup Backtest Configs
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
        max_turnover=0.2
    )
    
    # Determine frequency for metrics
    freq = 'weekly' if cfg.ENV.FINANCE.RESAMPLE_WEEKLY else 'daily'
    
    # 3. Run Strategies
    results = {}
    metrics = {}
    
    # --- Buy & Hold ---
    print("\n[1/5] Running Buy & Hold Strategy...")
    bh_strat = BuyAndHoldStrategy()
    results['Buy & Hold'] = run_backtest(bh_strat, env, bt_config)
    metrics['Buy & Hold'] = calculate_metrics(results['Buy & Hold'], freq=freq)
    
    # --- Markowitz ---
    print("\n[2/5] Running Markowitz Strategy...")
    # Gamma=1.0 is standard risk aversion. 
    # Adjust cost_coeff to be consistent or slightly higher if turnover is crazy.
    mark_strat = MarkowitzStrategy(risk_aversion=1.0, cost_coeff=0.001)
    results['Markowitz'] = run_backtest(mark_strat, env, bt_config)
    metrics['Markowitz'] = calculate_metrics(results['Markowitz'], freq=freq)
    
    # --- DMD (Linear Koopman) ---
    print("\n[3/5] Running DMD Strategy...")
    # Fit DMD on training data
    train_data = env.train_dataset.data
    dmd_strat = DMDStrategy(train_data, mpc_config)
    results['DMD-MPC'] = run_backtest(dmd_strat, env, bt_config)
    metrics['DMD-MPC'] = calculate_metrics(results['DMD-MPC'], freq=freq)
    
    # --- Koopman-MPC (Deep) - for each checkpoint ---
    print("\n[4/5] Running Koopman-MPC Strategies...")
    
    for label, ckpt_path in checkpoints_to_eval:
        print(f"  Evaluating {label} Checkpoint ({ckpt_path.name})...")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        
        # Re-create model to ensure clean state
        model = make_model(cfg, env.observation_size)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        
        strat_name = f"Koopman-MPC ({label})"
        kmpc_strat = KoopmanMPCStrategy(model, mpc_config)
        results[strat_name] = run_backtest(kmpc_strat, env, bt_config)
        metrics[strat_name] = calculate_metrics(results[strat_name], freq=freq)

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
        'Koopman-MPC (Best)': 'red',
        'Koopman-MPC (Last)': 'red' # Reuse red for whichever wins
    }
    styles = {
        'Buy & Hold': '--',
        'Markowitz': '-',
        'DMD-MPC': '-',
        'Koopman-MPC (Best)': '-',
        'Koopman-MPC (Last)': '-'
    }
    
    # Determine winner between Best and Last
    kmpc_best_val = results.get('Koopman-MPC (Best)', {}).get('portfolio_value', pd.Series([0])).iloc[-1] if 'Koopman-MPC (Best)' in results else -1
    kmpc_last_val = results.get('Koopman-MPC (Last)', {}).get('portfolio_value', pd.Series([0])).iloc[-1] if 'Koopman-MPC (Last)' in results else -1
    
    winner = None
    if kmpc_best_val > kmpc_last_val:
        winner = 'Koopman-MPC (Best)'
        loser = 'Koopman-MPC (Last)'
    else:
        winner = 'Koopman-MPC (Last)'
        loser = 'Koopman-MPC (Best)'
        
    print(f"\nPlotting winner: {winner} (End Value: ${max(kmpc_best_val, kmpc_last_val):.2f})")
    print(f"Skipping loser: {loser} (End Value: ${min(kmpc_best_val, kmpc_last_val):.2f})")
    
    # Use dates from one of the results
    dates = pd.to_datetime(results['Buy & Hold']['date'])
    
    for name, df in results.items():
        # Skip the loser
        if name == loser:
            continue
            
        c = colors.get(name, 'black')
        s = styles.get(name, '-')
        
        # Rename winner for legend if desired, or keep specific name
        label_name = name
        if name == winner:
            label_name = "Koopman-MPC" # Generic name for the plot
            
        plt.plot(dates, df['portfolio_value'], 
                 label=f"{label_name} (Sharpe: {metrics[name]['Sharpe Ratio']:.2f})",
                 color=c, linestyle=s, linewidth=1.5)
    
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
