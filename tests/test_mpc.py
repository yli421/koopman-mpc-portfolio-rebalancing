import pytest
import numpy as np
import cvxpy as cp
from mpc import solve_mpc_log_utility, MPCConfig

def test_mpc_feasibility():
    """Test that MPC returns feasible weights summing to 1."""
    N = 5
    H = 3
    current_weights = np.ones(N) / N
    predicted_log_returns = np.zeros((H, N)) # Flat returns
    
    config = MPCConfig(horizon=H, cost_coeff=0.0)
    
    w_opt, info = solve_mpc_log_utility(current_weights, predicted_log_returns, config)
    
    assert info["status"] == "optimal"
    assert w_opt.shape == (H, N)
    
    # Check constraints
    for t in range(H):
        assert np.isclose(np.sum(w_opt[t]), 1.0)
        assert np.all(w_opt[t] >= -1e-5) # Non-negative

def test_mpc_preference():
    """Test that MPC prefers the asset with highest return."""
    N = 2
    H = 1
    current_weights = np.array([0.5, 0.5])
    
    # Asset 0 has high return, Asset 1 has low return
    predicted_log_returns = np.array([[0.1, 0.0]])
    
    config = MPCConfig(horizon=H, cost_coeff=0.0)
    w_opt, _ = solve_mpc_log_utility(current_weights, predicted_log_returns, config)
    
    # Should shift towards Asset 0
    assert w_opt[0, 0] > 0.5
    assert w_opt[0, 1] < 0.5

def test_transaction_costs():
    """Test that high transaction costs prevent rebalancing."""
    N = 2
    H = 1
    current_weights = np.array([1.0, 0.0])
    
    # Asset 1 has slightly better return, but not enough to justify cost
    predicted_log_returns = np.array([[0.0, 0.01]])
    
    # High cost
    config = MPCConfig(horizon=H, cost_coeff=10.0)
    w_opt, _ = solve_mpc_log_utility(current_weights, predicted_log_returns, config)
    
    # Should stay at [1.0, 0.0] roughly
    assert np.allclose(w_opt[0], current_weights, atol=1e-2)

