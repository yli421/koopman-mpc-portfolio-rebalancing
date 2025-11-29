"""
Data module for dynamical systems environments.

Provides PyTorch-based implementations of various nonlinear dynamical systems
structured as environments.
"""

from abc import ABC, abstractmethod
from typing import Optional, Callable
import torch
from config import Config

import yfinance as yf
import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Base Environment Classes
# ---------------------------------------------------------------------------


class Env(ABC):
    """Base Environment class for dynamical systems."""

    def __init__(self, cfg: Optional[Config] = None):
        self.cfg = cfg

    @abstractmethod
    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Reset environment to initial state.
        
        Args:
            rng: Random number generator for reproducibility
            
        Returns:
            Initial state tensor
        """
        pass

    @abstractmethod
    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Take one step in the environment.
        
        Args:
            state: Current state tensor
            action: Optional action tensor (for controlled systems)
            
        Returns:
            Next state tensor
        """
        pass

    @property
    def observation_size(self) -> int:
        """Dimensionality of the state space."""
        rng = torch.Generator()
        rng.manual_seed(0)
        reset_state = self.unwrapped.reset(rng)
        return reset_state.shape[-1]

    @property
    def action_size(self) -> int:
        """Dimensionality of the action space."""
        return 0

    @property
    def unwrapped(self) -> 'Env':
        """Return the unwrapped environment."""
        return self


class Wrapper(Env):
    """Base Wrapper class for environment modifications."""

    def __init__(self, env: Env):
        super().__init__(cfg=None)
        self.env = env

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        return self.env.reset(rng)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        return self.env.step(state, action)

    @property
    def observation_size(self) -> int:
        return self.env.observation_size

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def unwrapped(self) -> Env:
        return self.env.unwrapped


class VectorWrapper(Wrapper):
    """Wrapper for vectorized/batched environment operations."""

    def __init__(self, env: Env, batch_size: int):
        super().__init__(env)
        self.batch_size = batch_size

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        """Reset multiple environments in parallel with independent random seeds.
        
        Mimics JAX's jax.random.split behavior by creating independent generators
        for each environment in the batch to ensure diverse initial states.
        
        Args:
            rng: Random number generator (if None, creates a new one)
            
        Returns:
            Batch of initial states with shape [batch_size, state_dim]
        """
        if rng is None:
            rng = torch.Generator()
        
        # Create independent generators for each environment (like jax.random.split)
        base_seed = rng.initial_seed()
        states = []
        for i in range(self.batch_size):
            env_rng = torch.Generator().manual_seed(base_seed + i)
            states.append(self.env.reset(env_rng))
        return torch.stack(states, dim=0)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Step multiple environments in parallel.
        
        Args:
            state: Batch of states with shape [batch_size, state_dim]
            action: Optional batch of actions
            
        Returns:
            Batch of next states with shape [batch_size, state_dim]
        """
        if action is None:
            return torch.vmap(lambda s: self.env.step(s, None))(state)
        else:
            return torch.vmap(lambda s, a: self.env.step(s, a))(state, action)
    
    def generate_sequence_batch(
        self, 
        rng: Optional[torch.Generator] = None,
        window_length: int = 10,
    ) -> torch.Tensor:
        """Generate a batch of sequence windows using vectorized operations.
        
        Args:
            rng: Random number generator
            window_length: Length of sequence (T steps after initial state)
            
        Returns:
            Batch of sequences with shape [batch_size, window_length+1, state_dim]
            Each sequence contains [x_t, x_{t+1}, ..., x_{t+T}]
        """
        init_states = self.reset(rng)  # [batch_size, state_dim]
        
        # Vectorized sequence generation: use generate_trajectory with batched step
        # This is much faster than the Python loop
        trajectories = generate_trajectory(
            self.step,
            init_states,
            length=window_length
        )  # [window_length, batch_size, state_dim]
        
        # Stack initial states with trajectories: [batch_size, window_length+1, state_dim]
        sequences = torch.cat([
            init_states.unsqueeze(0),  # [1, batch_size, state_dim]
            trajectories
        ], dim=0)  # [window_length+1, batch_size, state_dim]
        
        # Transpose to [batch_size, window_length+1, state_dim]
        return sequences.transpose(0, 1)


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def integrate_euler(
    x: torch.Tensor,
    u: Optional[torch.Tensor],
    dt: float,
    dynamics_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
) -> torch.Tensor:
    """Euler integration step for ODE.
    
    Args:
        x: Current state
        u: Optional control input
        dt: Time step
        dynamics_fn: Function computing state derivatives
        
    Returns:
        Next state using Euler integration
    """
    return x + dt * dynamics_fn(x, u)


def integrate_rk4(
    x: torch.Tensor,
    u: Optional[torch.Tensor],
    dt: float,
    dynamics_fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor]
) -> torch.Tensor:
    """Fourth-order Runge-Kutta (RK4) integration step for ODE.
    
    The RK4 method is a higher-order numerical integrator that provides
    better accuracy than Euler integration by using a weighted average
    of four slope estimates.
    
    Args:
        x: Current state
        u: Optional control input
        dt: Time step
        dynamics_fn: Function computing state derivatives
        
    Returns:
        Next state using RK4 integration
    """
    k1 = dynamics_fn(x, u)
    k2 = dynamics_fn(x + 0.5 * dt * k1, u)
    k3 = dynamics_fn(x + 0.5 * dt * k2, u)
    k4 = dynamics_fn(x + dt * k3, u)
    
    return x + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)


def generate_trajectory(
    env_step: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    init_state: torch.Tensor,
    length: Optional[int] = None,
    rng: Optional[torch.Generator] = None,
    actions: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Generate a trajectory by repeatedly applying environment step.
    
    Args:
        env_step: Function that takes state (and optionally action) and returns next state
        init_state: Initial state tensor
        length: Number of steps (required if actions is None)
        rng: Random number generator (unused, for API compatibility)
        actions: Optional sequence of actions with shape [length, action_dim]
        
    Returns:
        Trajectory tensor with shape [length, state_dim] or [length, batch_size, state_dim]
    """
    if actions is None:
        assert length is not None, "Must provide either length or actions"
        states = []
        state = init_state
        for _ in range(length):
            state = env_step(state)
            states.append(state)
        return torch.stack(states, dim=0)
    else:
        states = []
        state = init_state
        for action in actions:
            state = env_step(state, action)
            states.append(state)
        return torch.stack(states, dim=0)


def generate_sequence_window(
    env_step: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
    init_state: torch.Tensor,
    window_length: int,
) -> torch.Tensor:
    """Generate a sequence window including the initial state.
    
    Args:
        env_step: Function that takes state (and optionally action) and returns next state
        init_state: Initial state tensor
        window_length: Length of sequence window (T+1 total states including initial)
        
    Returns:
        Sequence tensor with shape [window_length+1, state_dim] or [window_length+1, batch_size, state_dim]
        This includes x_t, x_{t+1}, ..., x_{t+T}
    """
    states = [init_state]
    state = init_state
    for _ in range(window_length):
        state = env_step(state)
        states.append(state)
    return torch.stack(states, dim=0)


# ---------------------------------------------------------------------------
# Dynamical System Environments
# ---------------------------------------------------------------------------


class Pendulum(Env):
    """Pendulum model - freely swinging pole.
    
    State: [angle, angular_velocity]
    The angle x1 is measured in radians from the downward vertical position.
    Initial conditions sampled uniformly from [-π, π] × [-2, 2].
    
    Dynamics:
        dot(x1) = x2
        dot(x2) = -(g/L) * sin(x1)
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.g_over_l = 9.81 / 1.0
        self.dt = cfg.ENV.PENDULUM.DT

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        x1 = torch.empty(1).uniform_(-torch.pi, torch.pi, generator=rng)
        x2 = torch.empty(1).uniform_(-2.0, 2.0, generator=rng)
        return torch.tensor([x1.item(), x2.item()], dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            x1, x2 = state[0], state[1]
            dx1 = x2
            dx2 = -self.g_over_l * torch.sin(x1)
            return torch.stack([dx1, dx2])

        return integrate_rk4(state, None, self.dt, dynamics_fn)


class Duffing(Env):
    """Duffing Oscillator - damped and force-driven particle model.
    
    Nonlinear second-order ODE: x'' = x - x^3
    
    Admits two center points at (x, x') = (±1, 0) and an unstable fixed point at origin.
    Initial conditions sampled uniformly from [-2, 2] × [-1, 1].
    
    Dynamics:
        dot(x1) = x2
        dot(x2) = x1 - x1^3
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.dt = cfg.ENV.DUFFING.DT

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        x1 = torch.empty(1).uniform_(-1.5, 1.5, generator=rng)
        x2 = torch.empty(1).uniform_(-1.0, 1.0, generator=rng)
        return torch.tensor([x1.item(), x2.item()], dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            x1, x2 = state[0], state[1]
            dx1 = x2
            dx2 = x1 - x1**3
            return torch.stack([dx1, dx2])

        return integrate_rk4(state, None, self.dt, dynamics_fn)


class LotkaVolterra(Env):
    """Lotka-Volterra predator-prey model.
    
    Models population dynamics with predator-prey interactions.
    State: [prey_population, predator_population]
    
    Dynamics:
        dot(x1) = alpha * x1 - beta * x1 * x2
        dot(x2) = delta * x1 * x2 - gamma * x2
    
    Parameters: alpha = beta = gamma = delta = 0.2
    Fixed points: origin and center at (gamma/delta, alpha/beta) = (1, 1)
    Initial conditions sampled uniformly from [0.02, 3.0] × [0.02, 3.0]
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.alpha = 0.2
        self.beta = 0.2
        self.gamma = 0.2
        self.delta = 0.2
        self.dt = cfg.ENV.LOTKA_VOLTERRA.DT

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        x1 = torch.empty(1).uniform_(0.02, 3.0, generator=rng)
        x2 = torch.empty(1).uniform_(0.02, 3.0, generator=rng)
        return torch.tensor([x1.item(), x2.item()], dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            prey, predator = state[0], state[1]
            dx1 = self.alpha * prey - self.beta * prey * predator
            dx2 = self.delta * prey * predator - self.gamma * predator
            return torch.stack([dx1, dx2])

        return integrate_rk4(state, None, self.dt, dynamics_fn)


class Lorenz63(Env):
    """Lorenz 63 system - chaotic three-dimensional system.
    
    Famous for the "butterfly effect" and sensitivity to initial conditions.
    Exhibits chaotic behavior with strange attractor.
    
    Dynamics:
        dot(x1) = sigma * (x2 - x1)
        dot(x2) = x1 * (rho - x3) - x2
        dot(x3) = x1 * x2 - beta * x3
    
    Standard parameters: sigma=10, rho=28, beta=8/3 (Lorenz 1963)
    Initial conditions: perturbations around (0, 1, 1.05) with std=1.0
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.sigma = 10.0
        self.rho = 28.0
        self.beta = 8.0 / 3.0
        self.dt = cfg.ENV.LORENZ63.DT

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        base_point = torch.tensor([0.0, 1.0, 1.05], dtype=torch.float32)
        noise = torch.randn(3, generator=rng, dtype=torch.float32)
        return base_point + noise

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            x, y, z = state[0], state[1], state[2]
            dx = self.sigma * (y - x)
            dy = x * (self.rho - z) - y
            dz = x * y - self.beta * z
            return torch.stack([dx, dy, dz])

        return integrate_rk4(state, None, self.dt, dynamics_fn)


class Parabolic(Env):
    """Parabolic Attractor - two-dimensional system with parabolic manifold.
    
    Dynamics admit solution asymptotically attracted to x2 = x1^2 for lambda < mu < 0.
    
    Dynamics:
        dot(x1) = mu * x1
        dot(x2) = lambda * (x2 - x1^2)
    
    The Koopman embedding z = [x1, x2, x1^2] has globally linear dynamics:
        dot(z) = [mu*z1, lambda*z2 - lambda*z3, 2*mu*z3]
    
    Parameters: lambda=-1.0, mu=-0.1
    Initial conditions sampled uniformly from [-1, 1] × [-1, 1]
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        self.const_lambda = cfg.ENV.PARABOLIC.LAMBDA
        self.const_mu = cfg.ENV.PARABOLIC.MU
        self.dt = cfg.ENV.PARABOLIC.DT

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        x1 = torch.empty(1).uniform_(-1.0, 1.0, generator=rng)
        x2 = torch.empty(1).uniform_(-1.0, 1.0, generator=rng)
        return torch.tensor([x1.item(), x2.item()], dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            x1, x2 = state[0], state[1]
            dx1 = self.const_mu * x1
            dx2 = self.const_lambda * (x2 - x1**2)
            return torch.stack([dx1, dx2])

        return integrate_rk4(state, None, self.dt, dynamics_fn)


# ---------------------------------------------------------------------------
# Lyapunov Multi-Attractor Environment (from Koopman_learning.ipynb)
# ---------------------------------------------------------------------------


class LyapunovMultiAttractor(Env):
    """Nonlinear 2D system with many exponentially stable equilibria.
    
    Implements the vector field described in the notebook, with equilibria at
    a fixed set of 13 points and dynamics built from Gaussian bump functions.
    """

    def __init__(self, cfg: Config):
        super().__init__(cfg)
        # Configurable parameters
        # Defaults chosen to match the notebook example
        self.dt = getattr(cfg.ENV, 'LYAPUNOV', None).DT if hasattr(cfg.ENV, 'LYAPUNOV') else 0.05
        self.sigma = getattr(cfg.ENV, 'LYAPUNOV', None).SIGMA if hasattr(cfg.ENV, 'LYAPUNOV') else 0.5

        # Stable points (equilibria)
        self.points = torch.tensor([
            [-1.0, -1.0], [ 1.0, -1.0], [-1.0,  1.0], [ 1.0,  1.0],
            [ 0.0,  0.0],
            [-1.0, -2.0], [ 1.0, -2.0], [-1.0,  2.0], [ 1.0,  2.0],
            [-2.0, -1.0], [ 2.0, -1.0], [-2.0,  1.0], [ 2.0,  1.0],
        ], dtype=torch.float32)

        self._sigma2 = float(self.sigma) * float(self.sigma)

    @property
    def action_size(self) -> int:
        return 0

    def reset(self, rng: Optional[torch.Generator] = None) -> torch.Tensor:
        if rng is None:
            rng = torch.Generator()
        x1 = torch.empty(1).uniform_(-2.5, 2.5, generator=rng)
        x2 = torch.empty(1).uniform_(-2.5, 2.5, generator=rng)
        return torch.tensor([x1.item(), x2.item()], dtype=torch.float32)

    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        sigma2 = self._sigma2

        def dynamics_fn(state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
            # Vectorized over all equilibrium points
            diff = state.unsqueeze(0) - self.points  # [M, 2]
            r2 = (diff * diff).sum(dim=1)  # [M]
            normx2 = torch.dot(state, state)

            # First term: - 2/sigma^2 * sum_i diff_i * (||x||^2 * exp(-||x-p_i||^2 / sigma^2))
            psi1 = normx2 * torch.exp(-r2 / sigma2)  # [M]
            term1 = (-2.0 / sigma2) * (psi1.unsqueeze(1) * diff).sum(dim=0)  # [2]

            # Second term: -sum_i diff_i * exp(-||x-p_i||^2 / sigma^2)
            psi2 = torch.exp(-r2 / sigma2)  # [M]
            term2 = -(psi2.unsqueeze(1) * diff).sum(dim=0)  # [2]

            return term1 + term2

        return integrate_rk4(state, None, self.dt, dynamics_fn)


# ---------------------------------------------------------------------------
# Financial Data Environment (NYSE(N))
# ---------------------------------------------------------------------------


class FinanceEnvironment:
    """
    NYSE(N) Environment for Koopman Training.
    State: Price Relatives (Daily Returns + 1) - from paper J. Li et al., "DMD for Online Portfolio Selection".
    """
    def __init__(self, cfg):
        self.cfg = cfg
        self.tickers = cfg.ENV.FINANCE.TICKERS
        self.start = cfg.ENV.FINANCE.START_DATE
        self.end = cfg.ENV.FINANCE.END_DATE
        
        print(f"[FinanceEnv] Fetching NYSE(N) data ({self.start} to {self.end})...")
        
        # 1. Download Data 
        raw_df = yf.download(
            self.tickers, 
            start=self.start, 
            end=self.end, 
            interval="1d",
            auto_adjust=True, # handles stock splits/dividends over 26 years 
            progress=False
        )['Close']
        
        # 2. Data Cleaning
        # Drop columns that have too much missing data 
        # Threshold: Must have data for at least 90% of the timeframe
        valid_cols = raw_df.columns[raw_df.notna().sum() > len(raw_df) * 0.9]
        self.clean_df = raw_df[valid_cols].ffill().bfill()
        
        dropped = set(self.tickers) - set(valid_cols)
        if dropped:
            print(f"[FinanceEnv] Warning: Dropped {len(dropped)} tickers due to missing history: {dropped}")
            
        # 3. Compute Price Relatives, x_t = Price_t / Price_{t-1}
        self.price_relatives = self.clean_df.pct_change().fillna(0).values + 1.0
        
        # 4. To Tensor
        self.data = torch.tensor(self.price_relatives, dtype=torch.float32)
        self.num_timesteps, self.num_assets = self.data.shape
        self.observation_size = self.num_assets
        
        # 5. Time Indexing (Critical for sequential training)
        if cfg.ENV.FINANCE.INCLUDE_TIME_INDEX:
            self.observation_size += 1
            t_idx = torch.linspace(0, 1, self.num_timesteps).unsqueeze(1)
            self.data = torch.cat([self.data, t_idx], dim=1)
            
        print(f"[FinanceEnv] Final Dataset: {self.num_timesteps} days x {self.num_assets} stocks")

    def reset(self, rng: torch.Generator) -> torch.Tensor:
        # Return a random day from history (excluding last day)
        idx = torch.randint(0, self.num_timesteps - 1, (1,), generator=rng).item()
        return self.data[idx]
    
    def step(self, state: torch.Tensor, action: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Return the next day's data.
        Uses F.embedding for vmap-safe indexing.
        """
        # 1. Ensure the data table is on the same device as the incoming state (CPU vs GPU)
        if self.data.device != state.device:
            self.data = self.data.to(state.device)

        # 2. Recover integer time index from normalized feature
        t_norm = state[..., -1]
        t_idx = (t_norm * (self.num_timesteps - 1)).long()
        
        # 3. Get next step index (clamp to avoid out-of-bounds)
        next_t_idx = torch.clamp(t_idx + 1, max=self.num_timesteps - 1)
        
        # 4. Use embedding lookup instead of self.data[next_t_idx]
        # This prevents the "vmap .item()" error
        return torch.nn.functional.embedding(next_t_idx, self.data)
        
    def get_full_trajectory(self):
        return self.data
    

# ---------------------------------------------------------------------------
# Registry and Factory
# ---------------------------------------------------------------------------


_ENV_REGISTRY = {
    "pendulum": Pendulum,
    "duffing": Duffing,
    "lotka_volterra": LotkaVolterra,
    "lorenz63": Lorenz63,
    "parabolic": Parabolic,
    "lyapunov": LyapunovMultiAttractor,
    "finance": FinanceEnvironment,
}


def make_env(cfg: Config) -> Env:
    """Factory function to create environment from configuration.
    
    Args:
        cfg: Configuration object with ENV.ENV_NAME specifying the system
        
    Returns:
        Environment instance
        
    Raises:
        ValueError: If ENV_NAME is not in registry
    """
    env_name = cfg.ENV.ENV_NAME
    if env_name not in _ENV_REGISTRY:
        raise ValueError(
            f"Unknown environment '{env_name}'. "
            f"Available: {list(_ENV_REGISTRY.keys())}"
        )
    return _ENV_REGISTRY[env_name](cfg)