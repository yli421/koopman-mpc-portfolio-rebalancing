"""
Baseline Strategies for Portfolio Rebalancing.

1. Markowitz (Mean-Variance):
   - Uses rolling window to estimate mean returns (mu) and covariance (Sigma).
   - Solves standard QP: max w^T mu - gamma * w^T Sigma w - costs

2. DMD (Dynamic Mode Decomposition):
   - "Linear" Koopman baseline.
   - Fits a linear operator K on the training data: Y_{t+1} = K Y_t.
   - Predicts future returns by linear rollout.
   - Uses the same MPC solver as Koopman-MPC.
"""

import numpy as np
import cvxpy as cp
from scipy.linalg import pinv
import torch

from backtest import Strategy
from mpc import solve_mpc_log_utility, MPCConfig, solve_mpc_mean_variance
from data_finance import FinanceEnv

class MarkowitzStrategy(Strategy):
    """
    Classic Mean-Variance Optimization.
    Rebalances based on rolling window estimates of Mean and Covariance.
    """
    def __init__(
        self, 
        risk_aversion: float = 1.0, 
        cost_coeff: float = 0.001, 
        allow_short: bool = False
    ):
        self.risk_aversion = risk_aversion
        self.cost_coeff = cost_coeff
        self.allow_short = allow_short
        
        # We reuse the MPC config structure for the solver
        self.mpc_config = MPCConfig(
            horizon=1, # Markowitz is typically single-period
            gamma=risk_aversion,
            cost_coeff=cost_coeff,
            allow_short=allow_short,
            solver="ECOS"
        )

    def rebalance(self, t, current_weights, env: FinanceEnv, lookback_window: int = 60) -> np.ndarray:
        # 1. Get Historical Data (Lookback)
        # We need the RAW log-returns, not embedded.
        # env.test_dataset contains embedded data.
        # However, we can reconstruct the history from the "dates" indices or 
        # just assume we have access to the full return history.
        # For this implementation, we will extract the history from the dataset tensor.
        
        # NOTE: accessing env.test_dataset.data directly.
        # t is the index in the test set.
        # We need [t - lookback : t]
        
        # Since we might be at t=0, we ideally need training data history.
        # But for simplicity, we'll start using window from what's available or 
        # assume a warm-up period.
        
        # Let's grab the actual returns vector from the environment wrapper if possible,
        # or reconstructing it.
        # The env has `extract_current_returns` which gets the most recent return from embedding.
        
        # Get all available test data up to t
        # shape [t+1, obs_size]
        past_data = env.test_dataset.data[:t+1].to(env.test_dataset.data.device)
        
        # Extract returns: [t+1, n_assets]
        past_returns_std = env.extract_current_returns(past_data)
        past_returns = env.destandardize_returns(past_returns_std).cpu().numpy()
        
        if len(past_returns) < 5:
            # Not enough data for covariance, stick to current weights or 1/N
            return current_weights
            
        # Use only the last `lookback_window` points
        window = past_returns[-lookback_window:]
        
        # 2. Estimate mu and Sigma
        mu = np.mean(window, axis=0)
        sigma = np.cov(window, rowvar=False)
        
        # Add regularization to Sigma if it's ill-conditioned
        sigma += np.eye(len(mu)) * 1e-6
        
        # 3. Solve Optimization
        # We treat the single-period prediction as "predicted_log_returns" for the signature
        # But actually we need a dedicated MV solver.
        
        # We implemented solve_mpc_mean_variance in mpc.py but it takes predicted sequences.
        # Let's prepare inputs:
        # Horizon = 1
        mu_seq = mu.reshape(1, -1)
        
        w_opt, info = solve_mpc_mean_variance(
            current_weights,
            mu_seq,
            sigma,
            self.mpc_config
        )
        
        return w_opt[0]


class DMDStrategy(Strategy):
    """
    Dynamic Mode Decomposition (Linear Koopman) Strategy.
    
    1. Fits Linear operator K on Training Data: Y' = K Y
    2. At test time t, predicts Y_{t+1}...Y_{t+H} using K.
    3. Feeds predictions to the same MPC solver.
    """
    def __init__(self, train_data: torch.Tensor, mpc_config: MPCConfig):
        """
        Args:
            train_data: Tensor of shape [N, obs_size] from training set.
            mpc_config: Configuration for the MPC solver.
        """
        self.mpc_config = mpc_config
        self.K = self._fit_dmd(train_data.cpu().numpy())
        self.n_assets = None # Will be set during fit or rebalance
        
    def _fit_dmd(self, data: np.ndarray) -> np.ndarray:
        """
        Fit linear operator K minimizing ||Y' - Y K||_F
        Solution: K = (Y^dagger Y')^T  (assuming row vectors x_t)
        Or standard: X' = A X  => A = X' X^dagger
        
        Here data is [samples, features].
        X  = data[:-1].T  [features, samples-1]
        X' = data[1:].T   [features, samples-1]
        
        K_matrix (A) s.t. x_{t+1} = A x_t
        """
        X = data[:-1].T
        X_prime = data[1:].T
        
        # Compute Pseudo-inverse of X
        # A = X' * pinv(X)
        K_matrix = X_prime @ pinv(X)
        return K_matrix

    def rebalance(self, t, current_weights, env: FinanceEnv, lookback_window: int = 60) -> np.ndarray:
        # 1. Get current state Y_t
        y_t = env.test_dataset.data[t].cpu().numpy() # [obs_size]
        
        # 2. Rollout linear dynamics
        H = self.mpc_config.horizon
        preds = []
        curr = y_t
        
        for _ in range(H):
            # x_{k+1} = A x_k
            # But y_t is a row vector in dataset? Yes, shape [obs_size]
            # Our K is designed for column vectors x_t
            # so x_{k+1} = K @ x_k
            
            curr_col = curr.reshape(-1, 1)
            next_col = self.K @ curr_col
            curr = next_col.flatten()
            
            # Extract returns part (first n_assets)
            # We need to act on the full embedding state to propagate dynamics
            
            # Extract return for MPC
            # Note: We need to handle the fact that 'curr' is standardized embedding
            # extract_current_returns expects torch tensor
            curr_tensor = torch.from_numpy(curr).float().to(env.test_dataset.data.device)
            ret_std = env.extract_current_returns(curr_tensor)
            ret_real = env.destandardize_returns(ret_std)
            
            preds.append(ret_real.cpu().numpy())
            
        pred_log_returns = np.array(preds) # [H, n_assets]
        
        # 3. Solve MPC
        new_weights, info = solve_mpc_log_utility(
            current_weights,
            pred_log_returns,
            self.mpc_config
        )
        
        return new_weights[0]

