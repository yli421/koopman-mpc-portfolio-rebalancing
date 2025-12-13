"""
Backtesting module for Koopman-MPC Portfolio Rebalancing.

This module implements the backtesting engine and various strategies including
the proposed Koopman-MPC approach and baselines (Buy & Hold, Markowitz).
"""

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Union
from abc import ABC, abstractmethod
from tqdm import tqdm

from config import Config
from mpc import solve_mpc_log_utility, MPCConfig
from model import KoopmanMachine
from data_finance import FinanceEnv

@dataclass
class BacktestConfig:
    """Configuration for backtesting."""
    initial_capital: float = 10000.0
    horizon: int = 5  # MPC prediction horizon
    rebalance_freq: int = 1  # Rebalance every N days
    cost_coeff: float = 0.001  # 10bps transaction cost
    risk_free_rate: float = 0.0  # Daily risk-free rate
    allow_short: bool = False

class Strategy(ABC):
    """Abstract base class for trading strategies."""
    
    @abstractmethod
    def rebalance(
        self,
        t: int,
        current_weights: np.ndarray,
        env: FinanceEnv,
        lookback_window: int = 60
    ) -> np.ndarray:
        """
        Determine new portfolio weights.
        
        Args:
            t: Current time index in the test set
            current_weights: Current portfolio weights [n_assets]
            env: Finance environment (provides access to history)
            lookback_window: Number of past days to use for estimation
            
        Returns:
            New weights [n_assets]
        """
        pass

class BuyAndHoldStrategy(Strategy):
    """Buy and Hold strategy (Equal Weight)."""
    
    def rebalance(self, t, current_weights, env, lookback_window=60):
        # Only rebalance at the very beginning to equal weights
        if t == 0:
            n_assets = env.n_assets
            return np.ones(n_assets) / n_assets
        return current_weights

class MarkowitzStrategy(Strategy):
    """Rolling Mean-Variance Optimization Strategy."""
    
    def __init__(self, risk_aversion: float = 1.0, cost_coeff: float = 0.001):
        self.risk_aversion = risk_aversion
        self.cost_coeff = cost_coeff
        
    def rebalance(self, t, current_weights, env, lookback_window=60):
        # Need historical log returns up to t
        # Access underlying data from test_dataset
        # Note: This accesses "future" data relative to start of test, 
        # but "past" relative to current time t.
        
        # In FinanceEnv, data is pre-embedded. 
        # We need raw log returns. 
        # We can reconstruct them or use a helper if available.
        # For simplicity, we'll assume we can access the full log_returns array 
        # and slice it up to t.
        
        # Hack: recover log returns from embedded data for simplicity
        # or use the fact that we have the full dataset.
        
        # Let's use a simpler approach: 
        # We assume the env has a way to get past returns.
        # For this implementation, we will pass the full returns array to the backtester
        # and slice it here.
        pass # Implemented inside run_backtest context usually, but here we mock
        
        # Placeholder for now as Markowitz requires solving a QP similar to MPC
        # We can reuse the MPC solver with gamma > 0 and horizon = 1
        return current_weights 

class KoopmanMPCStrategy(Strategy):
    """Koopman-MPC Strategy."""
    
    def __init__(
        self, 
        model: KoopmanMachine, 
        mpc_config: MPCConfig,
        device: str = 'cpu'
    ):
        self.model = model
        self.mpc_config = mpc_config
        self.device = device
        
    def rebalance(self, t, current_weights, env, lookback_window=60):
        # Get current observation Y_t
        # env.test_dataset[t] returns (Y_t, Y_{t+1}) or sequence
        # We just need Y_t
        
        obs = env.test_dataset.data[t].to(self.device).unsqueeze(0) # [1, obs_size]
        
        # Forecast future log-returns using Koopman
        # Horizon H
        H = self.mpc_config.horizon
        
        with torch.no_grad():
            self.model.eval()
            # Rollout
            # We use every-step re-encoding if we had ground truth, but for control
            # we only have the current state. So we must do pure Koopman rollout
            # or use predicted states.
            
            # Encode current state
            z = self.model.encode(obs)
            
            pred_log_returns = []
            
            # First step: next return y_{t+1}
            # The model predicts x_{t+1} which contains y_{t+1}
            
            curr_z = z
            for _ in range(H):
                # Step latent
                curr_z = self.model.step_latent(curr_z)
                # Decode
                pred_obs = self.model.decode(curr_z)
                
                # Extract y_{t+k} (first n_assets elements)
                pred_y = env.extract_current_returns(pred_obs)
                
                # Destandardize
                pred_y_real = env.destandardize_returns(pred_y)
                
                pred_log_returns.append(pred_y_real.cpu().numpy().flatten())
                
            pred_log_returns = np.array(pred_log_returns) # [H, n_assets]
            
        # Solve MPC
        new_weights, info = solve_mpc_log_utility(
            current_weights, 
            pred_log_returns, 
            self.mpc_config
        )
        
        # We only apply the first step weights
        return new_weights[0]

def run_backtest(
    strategy: Strategy,
    env: FinanceEnv,
    config: BacktestConfig,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Run backtest loop.
    
    Args:
        strategy: Strategy instance
        env: Finance environment
        config: Backtest configuration
        
    Returns:
        DataFrame with daily metrics (portfolio value, returns, turnover)
    """
    n_steps = len(env.test_dataset) - config.horizon # Leave room for horizon
    n_assets = env.n_assets
    
    # Initialize state
    cash = config.initial_capital
    weights = np.zeros(n_assets)
    portfolio_value = config.initial_capital
    
    history = []
    
    # Initial Weights (1/N) to ensure feasibility of turnover constraints at t=0
    current_weights = np.ones(n_assets) / n_assets

    iter_range = range(0, n_steps, config.rebalance_freq)
    if verbose:
        iter_range = tqdm(iter_range, desc="Backtesting")
    
    # Pre-load real returns for efficiency
    # These are standardized in the dataset, need to destandardize
    all_data = env.test_dataset.data.to(env.test_dataset.data.device)
    all_returns_std = env.extract_current_returns(all_data)
    all_returns = env.destandardize_returns(all_returns_std).cpu().numpy()
    
    for t in iter_range:
        # 1. Rebalance Step
        target_weights = strategy.rebalance(t, current_weights, env)
        
        # 2. Calculate Transaction Costs
        # cost = coeff * sum(|w_new - w_old|) * Value
        turnover = np.sum(np.abs(target_weights - current_weights))
        cost = config.cost_coeff * turnover * portfolio_value
        
        # Update weights and value
        current_weights = target_weights
        portfolio_value -= cost
        
        # 3. Simulate Market Step (t -> t+1)
        # Realized return at t+1 (using data at t+1)
        port_ret = 0.0
        
        # Note: all_returns[t+1] is the return from t to t+1
        if t + 1 < len(all_returns):
            realized_log_ret = all_returns[t+1]
            realized_ret = np.exp(realized_log_ret) - 1.0
            
            # Portfolio return
            port_ret = np.sum(current_weights * realized_ret)
            
            # Update Value
            portfolio_value *= (1.0 + port_ret)
            
            # Update Weights (drift)
            # w_i(t+1) = w_i(t) * (1+r_i) / (1+r_p)
            current_weights = current_weights * (1.0 + realized_ret) / (1.0 + port_ret)
        
        # Record stats
        history.append({
            'date': env.test_dataset.dates[t],
            'portfolio_value': portfolio_value,
            'return': port_ret,
            'turnover': turnover,
            'cost': cost
        })
        
    return pd.DataFrame(history)

def calculate_metrics(df: pd.DataFrame) -> Dict:
    """Calculate Sharpe, Max Drawdown, Turnover."""
    if len(df) == 0:
        return {}
        
    # Returns
    returns = df['return'].values
    
    # Annualized Sharpe (assuming daily data, 252 days)
    mean_ret = np.mean(returns)
    std_ret = np.std(returns)
    sharpe = np.sqrt(252) * mean_ret / (std_ret + 1e-8)
    
    # Max Drawdown
    cum_returns = (1 + returns).cumprod()
    peak = np.maximum.accumulate(cum_returns)
    drawdown = (cum_returns - peak) / peak
    max_dd = np.min(drawdown)
    
    # Total Turnover
    avg_turnover = df['turnover'].mean()
    
    return {
        'Sharpe Ratio': sharpe,
        'Max Drawdown': max_dd,
        'Avg Turnover': avg_turnover,
        'Final Value': df['portfolio_value'].iloc[-1],
        'Total Return': (df['portfolio_value'].iloc[-1] / df['portfolio_value'].iloc[0]) - 1.0
    }

if __name__ == "__main__":
    # Example usage
    pass

