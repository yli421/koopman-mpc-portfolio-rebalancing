"""
Model Predictive Control (MPC) module for Portfolio Rebalancing.

This module implements the convex optimization logic to determine optimal
portfolio weights given predicted future returns.

Formulation based on IFT6162 Project PDF:
Objective: Maximize expected logarithmic growth (Kelly) or Mean-Variance,
           penalized by transaction costs.
"""

import cvxpy as cp
import numpy as np
from dataclasses import dataclass
from typing import Tuple, Optional, Dict

@dataclass
class MPCConfig:
    """Configuration for MPC solver."""
    horizon: int = 5
    gamma: float = 0.0  # Risk aversion (0.0 = log utility maximization)
    cost_coeff: float = 0.001  # Transaction cost coefficient (e.g. 10bps)
    max_turnover: float = 0.2  # Maximum turnover per step
    allow_short: bool = False
    solver: str = "ECOS"  # ECOS or SCS are good for these problems
    risk_free_rate: float = 0.0  # Daily risk-free rate (e.g., 0.0 for 0%, 1e-4 for ~2.5% annual)

def solve_mpc_log_utility(
    current_weights: np.ndarray,
    predicted_log_returns: np.ndarray,
    config: MPCConfig,
) -> Tuple[np.ndarray, Dict]:
    """
    Solve MPC using Log-Utility Maximization (Kelly Criterion) with Cash Asset.
    
    maximize sum_{t=1}^H [ log(w_t^T exp(y_t)) - cost * ||w_t - w_{t-1}||_1 ]
    subject to:
        sum(w_t) = 1
        w_t >= 0 (if no short)
    
    Args:
        current_weights: Current portfolio weights [n_assets + 1] (last is cash)
        predicted_log_returns: Predicted log-returns of risky assets [horizon, n_assets]
        config: MPC configuration
        
    Returns:
        optimal_weights: [horizon, n_assets + 1]
        info: Dictionary with solve status and value
    """
    H, N = predicted_log_returns.shape
    # N is number of risky assets. Total assets = N + 1 (last is cash)
    
    # Ensure current_weights matches N+1
    if len(current_weights) != N + 1:
        # If passed N weights, append 0 for cash (backward compatibility or correction)
        if len(current_weights) == N:
            current_weights = np.append(current_weights, 0.0)
        else:
            raise ValueError(f"Expected {N+1} weights (incl. cash), got {len(current_weights)}")

    # Convert log-returns to gross returns: R = exp(y)
    # This approximates the gross return factor for the asset.
    # Note: If y is small, exp(y) ~ 1 + y.
    # For log-utility, we maximize log(w^T R).
    risky_returns = np.exp(predicted_log_returns)
    
    # Add cash return column (constant risk-free rate)
    # Gross return for cash is exp(r_f) approx 1 + r_f
    cash_return = np.exp(config.risk_free_rate)
    cash_col = np.full((H, 1), cash_return)
    predicted_returns = np.hstack([risky_returns, cash_col]) # [H, N+1]
    
    # Variables: weights over horizon (including cash)
    # w[t] corresponds to weights held during period t (to capture return r_t)
    w = cp.Variable((H, N + 1))
    
    objective_terms = []
    constraints = []
    
    # Initial turnover cost: change from current_weights to w[0]
    # We pay costs to rebalance TO w[0] at the start of period 1
    # Cost is usually paid on risky asset turnover.
    # Let's assume standard L1 norm on full vector for simplicity as per common formulations,
    # or restrict to risky assets if desired. 
    # Since selling stock -> cash implies cost, and buying stock <- cash implies cost,
    # L1 norm on full vector counts the trade volume (times 2).
    delta_0 = cp.norm(w[0] - current_weights, 1)
    cost_0 = config.cost_coeff * delta_0
    
    # We subtract cost from the portfolio value before growth? 
    # Or just penalize the objective?
    # Standard approximation: penalize objective.
    
    # For t=0 to H-1
    for t in range(H):
        # Expected portfolio return for period t
        # ret_p = w[t]^T R[t]
        port_return = w[t] @ predicted_returns[t]
        
        # Log utility of wealth growth
        objective_terms.append(cp.log(port_return))
        
        # Constraints
        constraints.append(cp.sum(w[t]) == 1.0)
        
        if not config.allow_short:
            constraints.append(w[t] >= 0)
            
        # Turnover constraints/costs for subsequent steps
        if t > 0:
            delta_t = cp.norm(w[t] - w[t-1], 1)
            cost_t = config.cost_coeff * delta_t
            objective_terms.append(-cost_t)
            
            if config.max_turnover > 0:
                constraints.append(delta_t <= config.max_turnover)
    
    # Subtract initial cost
    objective_terms.append(-cost_0)
    if config.max_turnover > 0:
        constraints.append(delta_0 <= config.max_turnover)
        
    # Define problem
    objective = cp.Maximize(cp.sum(objective_terms))
    problem = cp.Problem(objective, constraints)
    
    # Solve
    try:
        problem.solve(solver=getattr(cp, config.solver), verbose=False)
    except cp.SolverError:
        # Fallback to SCS if ECOS fails
        problem.solve(solver=cp.SCS, verbose=False)
        
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        # Fallback: keep weights constant
        return np.tile(current_weights, (H, 1)), {"status": problem.status, "value": None}
        
    return w.value, {"status": problem.status, "value": problem.value}

def solve_mpc_mean_variance(
    current_weights: np.ndarray,
    predicted_log_returns: np.ndarray,
    cov_matrix: np.ndarray,
    config: MPCConfig,
) -> Tuple[np.ndarray, Dict]:
    """
    Solve MPC using Mean-Variance Optimization with Cash Asset.
    
    maximize sum_{t=1}^H [ w_t^T mu_t - gamma * w_t^T Sigma w_t - cost * ||w_t - w_{t-1}||_1 ]
    
    Args:
        current_weights: Current portfolio weights [n_assets + 1]
        predicted_log_returns: Predicted log-returns (used as mu) [horizon, n_assets]
        cov_matrix: Covariance matrix of RISKY returns [n_assets, n_assets]
        config: MPC configuration
        
    Returns:
        optimal_weights: [horizon, n_assets + 1]
    """
    H, N = predicted_log_returns.shape
    
    # Ensure current_weights matches N+1
    if len(current_weights) != N + 1:
        if len(current_weights) == N:
            current_weights = np.append(current_weights, 0.0)
        else:
            raise ValueError(f"Expected {N+1} weights, got {len(current_weights)}")
            
    # Variables (include cash)
    w = cp.Variable((H, N + 1))
    
    objective_terms = []
    constraints = []
    
    # Initial cost
    cost_0 = config.cost_coeff * cp.norm(w[0] - current_weights, 1)
    objective_terms.append(-cost_0)
    
    # Cash return (log approx)
    r_f = config.risk_free_rate
    
    for t in range(H):
        # Expected return: w^T mu
        # Use log-returns as approximation for mu
        # w[t] is [N+1], predicted_log_returns[t] is [N]
        # mu_full = [predicted... ; r_f]
        mu_risky = predicted_log_returns[t]
        
        # Split weights into risky and cash
        w_risky = w[t][:N]
        w_cash = w[t][N]
        
        ret_term = w_risky @ mu_risky + w_cash * r_f
        
        # Variance risk: w_risky^T Sigma w_risky
        # Cash has 0 variance
        risk_term = config.gamma * cp.quad_form(w_risky, cov_matrix)
        
        objective_terms.append(ret_term - risk_term)
        
        # Constraints
        constraints.append(cp.sum(w[t]) == 1.0)
        if not config.allow_short:
            constraints.append(w[t] >= 0)
            
        # Subsequent costs
        if t > 0:
            cost_t = config.cost_coeff * cp.norm(w[t] - w[t-1], 1)
            objective_terms.append(-cost_t)
            
    objective = cp.Maximize(cp.sum(objective_terms))
    problem = cp.Problem(objective, constraints)
    
    try:
        problem.solve(solver=getattr(cp, config.solver), verbose=False)
    except:
         problem.solve(solver=cp.SCS, verbose=False)
         
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        return np.tile(current_weights, (H, 1)), {"status": problem.status}
        
    return w.value, {"status": problem.status, "value": problem.value}
