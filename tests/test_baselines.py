import pytest
import numpy as np
import torch
from unittest.mock import MagicMock
from baselines import MarkowitzStrategy

class MockFinanceEnv:
    def __init__(self):
        self.n_assets = 2
        self.observation_size = 5
        self.test_dataset = MagicMock()
        # Create some data: 20 time steps, 5 features
        # Features 0-1 are returns
        self.test_dataset.data = torch.randn(20, 5) 
        self.test_dataset.data.to = MagicMock(return_value=self.test_dataset.data)
        
        # Mock returns extraction: just take first 2 cols
        self.extract_current_returns = lambda x: x[..., :2]
        # Mock destandardize: identity for simplicity or simple scaling
        self.destandardize_returns = lambda x: x 

def test_markowitz_strategy_initialization():
    strat = MarkowitzStrategy(risk_aversion=2.0, cost_coeff=0.01)
    assert strat.risk_aversion == 2.0
    assert strat.cost_coeff == 0.01
    assert strat.mpc_config.gamma == 2.0
    assert strat.mpc_config.horizon == 1

def test_markowitz_rebalance_not_enough_data():
    env = MockFinanceEnv()
    strat = MarkowitzStrategy()
    current_weights = np.array([0.5, 0.5])
    
    # t=2 means we have indices 0, 1, 2 (3 samples). Less than 5.
    new_weights = strat.rebalance(t=2, current_weights=current_weights, env=env)
    
    # Should return current weights
    assert np.allclose(new_weights, current_weights)

def test_markowitz_rebalance_optimization():
    env = MockFinanceEnv()
    # Create deterministic data where Asset 0 has higher return and lower variance
    # Asset 0: 0.1 constant
    # Asset 1: 0.0 constant
    data = torch.zeros(20, 5)
    data[:, 0] = 0.1 # Asset 0 return
    data[:, 1] = 0.0 # Asset 1 return
    env.test_dataset.data = data
    env.test_dataset.data.to = MagicMock(return_value=env.test_dataset.data)
    
    strat = MarkowitzStrategy(risk_aversion=1.0)
    current_weights = np.array([0.5, 0.5])
    
    # t=10 means enough data
    new_weights = strat.rebalance(t=10, current_weights=current_weights, env=env)
    
    # Should favor Asset 0
    assert new_weights[0] > 0.5
    assert new_weights[1] < 0.5
    assert np.isclose(np.sum(new_weights), 1.0)

