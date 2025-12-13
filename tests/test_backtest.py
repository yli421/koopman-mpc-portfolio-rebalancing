import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import MagicMock

from backtest import (
    run_backtest, 
    BacktestConfig, 
    BuyAndHoldStrategy, 
    calculate_metrics
)
from data_finance import FinanceEnv, FinanceDataset

class MockFinanceEnv:
    def __init__(self):
        self.n_assets = 2
        self.test_dataset = MagicMock()
        self.test_dataset.dates = pd.date_range("2021-01-01", periods=10)
        self.test_dataset.data = torch.randn(10, 10) # 10 samples, obs_size 10
        self.test_dataset.__len__.return_value = 10
        
        # Mock returns extraction
        self.extract_current_returns = lambda x: x[..., :2]
        self.destandardize_returns = lambda x: x * 0.01 # Small returns

def test_backtest_mechanics():
    """Test that backtest loop runs and produces dataframe."""
    env = MockFinanceEnv()
    config = BacktestConfig(horizon=2, initial_capital=1000.0)
    strategy = BuyAndHoldStrategy()
    
    results = run_backtest(strategy, env, config, verbose=False)
    
    assert len(results) == 8 # 10 - horizon
    assert 'portfolio_value' in results.columns
    assert 'turnover' in results.columns
    assert results['portfolio_value'].iloc[-1] != 1000.0 # Should change

def test_metrics():
    """Test metric calculation."""
    df = pd.DataFrame({
        'return': [0.01, -0.01, 0.02, 0.0],
        'turnover': [0.1, 0.0, 0.0, 0.0],
        'portfolio_value': [1010, 999.9, 1019.9, 1019.9]
    })
    
    metrics = calculate_metrics(df)
    
    assert 'Sharpe Ratio' in metrics
    assert 'Max Drawdown' in metrics
    assert metrics['Max Drawdown'] < 0

