"""
PyTorch implementation of Koopman Autoencoder models.

This module provides:
- MLPCoder: Multi-layer perceptron for encoding/decoding
- LISTA: Learned Iterative Soft-Thresholding Algorithm for sparse coding
- KoopmanMachine: Abstract base class for Koopman operator learning
- GenericKM: Standard Koopman autoencoder with MLP encoder
- LISTAKM: Koopman machine with LISTA sparse encoder
"""

from abc import ABC, abstractmethod
from typing import Optional, Dict, Tuple, List
import torch
import torch.nn as nn
from config import Config

try:
    from torchdiffeq import odeint
    HAS_TORCHDIFFEQ = True
except ImportError:
    HAS_TORCHDIFFEQ = False


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------


def shrink(x: torch.Tensor, threshold: float) -> torch.Tensor:
    """Soft thresholding operator (shrinkage). Used in LISTA.
    
    Args:
        x: Input tensor
        threshold: Threshold value for soft thresholding
        
    Returns:
        Shrunk tensor
    """
    return torch.sign(x) * torch.maximum(torch.abs(x) - threshold, torch.zeros_like(x))


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.
    
    Args:
        name: Activation name ('relu', 'tanh', 'gelu')
        
    Returns:
        Activation module
    """
    activations = {
        'relu': nn.ReLU(),
        'tanh': nn.Tanh(),
        'gelu': nn.GELU(),
    }
    if name not in activations:
        raise ValueError(f"Unknown activation '{name}'. Available: {list(activations.keys())}")
    return activations[name]


# ---------------------------------------------------------------------------
# Network Components
# ---------------------------------------------------------------------------


class MLPCoder(nn.Module):
    """Multi-layer perceptron for encoding or decoding.
    
    Args:
        input_size: Input dimension
        target_size: Output dimension
        hidden_layers: List of hidden layer sizes
        last_relu: Whether to apply ReLU to the output
        use_bias: Whether to use bias in linear layers
        activation: Activation function name
    """
    
    def __init__(
        self,
        input_size: int,
        target_size: int,
        hidden_layers: List[int],
        last_relu: bool = False,
        use_bias: bool = False,
        activation: str = 'relu'
    ):
        super().__init__()
        self.input_size = input_size
        self.target_size = target_size
        self.hidden_layers = hidden_layers
        self.last_relu = last_relu
        
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size, bias=use_bias))
            layers.append(get_activation(activation))
            prev_size = hidden_size
        
        layers.append(nn.Linear(prev_size, target_size, bias=use_bias))
        if last_relu:
            layers.append(nn.ReLU())
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Input tensor of shape [..., input_size]
            
        Returns:
            Output tensor of shape [..., target_size]
        """
        return self.network(x)


class LISTA(nn.Module):
    """Learned Iterative Soft-Thresholding Algorithm (LISTA) encoder.
    
    This module implements a LISTA-style encoder: an unrolled, fixed-depth
    approximation to sparse coding built from alternating affine transforms
    and an elementwise soft-thresholding nonlinearity.

    Canonical LISTA (Gregor & LeCun, 2010) uses a linear pre-activation
    z-affine map W_e x and shared "mutual-inhibition" matrix S, with the
    nonlinearity given by the soft-thresholding (shrinkage) operator
    T_λ(v)_i = sign(v_i) * max(|v_i| - λ, 0). The overall encoder is
    therefore nonlinear due to T_λ.
    
    Shapes (standard convention):
        x ∈ ℝ^{xdim},  z ∈ ℝ^{zdim}
        Dictionary W_d ∈ ℝ^{xdim × zdim}  (columns are atoms)
        Linear encoder W_e = (1/L) W_dᵀ ∈ ℝ^{zdim × xdim}
        Inhibition S = I - (1/L) W_dᵀ W_d ∈ ℝ^{zdim × zdim}

    Iterations:
        c = W_e x
        z^(0) = T_{α/L}(c)
        for k = 0..K-1:
            z^(k+1) = T_{α/L}(S z^(k) + c)
        return z^(K)

    Notes:
        • If `use_linear_encode=True`, the module uses the canonical linear
          pre-activation W_e x. If `False`, an MLP can be used to produce c;
          this yields a LISTA-style unrolled network rather than canonical LISTA.
        • L is a Lipschitz constant estimate (e.g., ≥ spectral norm of W_dᵀ W_d).
        • α controls sparsity; K is the number of unrolled iterations.

    Args:
        cfg: Configuration object.
        xdim: Input dimension.
        Wd_init: Initial dictionary matrix with shape [xdim, zdim].
    """
    
    def __init__(self, cfg: Config, xdim: int, Wd_init: torch.Tensor):
        super().__init__()
        self.cfg = cfg
        self.xdim = xdim
        self.zdim = cfg.MODEL.TARGET_SIZE
        self.num_loops = cfg.MODEL.ENCODER.LISTA.NUM_LOOPS
        self.alpha = cfg.MODEL.ENCODER.LISTA.ALPHA
        self.L = cfg.MODEL.ENCODER.LISTA.L
        self.use_linear_encode = cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER
        
        assert Wd_init.shape == (xdim, self.zdim), \
            f"Wd_init shape {Wd_init.shape} doesn't match expected ({xdim}, {self.zdim})"
        
        if self.use_linear_encode:
            self.We = nn.Linear(xdim, self.zdim, bias=False)
            # Initialize as (1/L) * Wd^T
            with torch.no_grad():
                self.We.weight.copy_((1.0 / self.L) * Wd_init.T)  # [zdim, xdim]
        else:
            self.We = MLPCoder(
                input_size=xdim,
                target_size=self.zdim,
                hidden_layers=cfg.MODEL.ENCODER.LAYERS,
                use_bias=cfg.MODEL.ENCODER.USE_BIAS,
                last_relu=cfg.MODEL.ENCODER.LAST_RELU,
                activation=cfg.MODEL.ENCODER.ACTIVATION,
            )
        
        S_init = torch.eye(self.zdim) - (1.0 / self.L) * (Wd_init.T @ Wd_init)
        self.S = nn.Parameter(S_init)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass: iterative soft-thresholding.
        
        Args:
            x: Input tensor of shape [..., xdim]
            
        Returns:
            Sparse codes of shape [..., zdim]
        """
        # Initial encoding
        nonsparse_code = self.We(x)
        
        # Initialize with soft-thresholding of initial encoding
        z = shrink(nonsparse_code, self.alpha / self.L)
        
        # Iterative refinement
        for _ in range(self.num_loops):
            z = shrink(z @ self.S + nonsparse_code, self.alpha / self.L)
        
        return z


# ---------------------------------------------------------------------------
# Koopman Machine Base Class
# ---------------------------------------------------------------------------

class KoopmanMachine(ABC, nn.Module):
    """Abstract base class for Koopman operator learning.
    
    The Koopman operator is a linear operator that provides a mathematical 
    framework for representing the dynamics of a nonlinear dynamical system (NLDS) 
    in terms of an infinite-dimensional linear operator. 
    Formally, the Koopman operator advances a measurement function forward in time 
    through the underlying system dynamics.
    
    This class provides the interface for learning Koopman representations.
    
    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space
    """
    
    def __init__(self, cfg: Config, observation_size: int):
        super().__init__()
        self.cfg = cfg
        self.observation_size = observation_size
        self.target_size = cfg.MODEL.TARGET_SIZE
        self.dt = None  # Will be set from environment config if needed
    
    @abstractmethod
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent space.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Latent codes of shape [..., target_size]
        """
        pass
    
    @abstractmethod
    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Decode latent representations to observation space.
        
        Args:
            y: Latent representations of shape [..., target_size]
            
        Returns:
            Reconstructed observations of shape [..., observation_size]
        """
        pass
    
    @abstractmethod
    def kmatrix(self) -> torch.Tensor:
        """Extract the learned Koopman matrix from parameters.
        
        Returns:
            Koopman matrix of shape [target_size, target_size]
        """
        pass
    
    def residual(self, x: torch.Tensor, nx: torch.Tensor) -> torch.Tensor:
        """Compute alignment loss between consecutive states in latent space.
        Determines how linearly aligned x & nx are in the latent space.
        
        Args:
            x: Current states of shape [..., observation_size]
            nx: Next states of shape [..., observation_size]
            
        Returns:
            Residual norms of shape [...]
        """
        y = self.encode(x)
        ny = self.encode(nx)
        kmat = self.kmatrix()
        return torch.norm(y @ kmat - ny, dim=-1)
    
    def reconstruction(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstruction via encode-decode.
        
        Args:
            x: shape [..., observation_size]
            
        Returns:
            Reconstructions of shape [..., observation_size]
        """
        return self.decode(self.encode(x))
    
    def sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L1 sparsity loss on latent codes.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Scalar sparsity loss
        """
        z = self.encode(x)
        return torch.norm(z, p=1, dim=-1).mean()
    
    def step_latent(self, y: torch.Tensor) -> torch.Tensor:
        """Step forward in latent space using Koopman matrix.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Next latent codes of shape [..., target_size]
        """
        kmat = self.kmatrix()
        return y @ kmat
    
    def step_env(self, x: torch.Tensor) -> torch.Tensor:
        """Predict next observation using Koopman dynamics.
        
        Args:
            x: Current observations of shape [..., observation_size]
            
        Returns:
            Predicted next observations of shape [..., observation_size]
        """
        y = self.encode(x)
        ny = self.step_latent(y)
        nx = self.decode(ny)
        return nx
    
    def koopman_ode_func(self, t: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        """ODE function for continuous-time Koopman dynamics: dz/dt = K @ z.
        
        Args:
            t: Time (scalar, unused but required by odeint)
            z: Latent state of shape [..., target_size]
            
        Returns:
            Time derivative dz/dt of shape [..., target_size]
        """
        kmat = self.kmatrix()
        # For linear dynamics: dz/dt = K @ z
        return z @ kmat
    
    def integrate_latent_ode(
        self, 
        z0: torch.Tensor, 
        t_span: torch.Tensor,
        method: str = 'dopri5'
    ) -> torch.Tensor:
        """Integrate Koopman dynamics from z0 over time points in t_span.
        
        Args:
            z0: Initial latent state of shape [batch_size, target_size]
            t_span: Time points of shape [num_steps+1] starting from 0
            method: Integration method ('dopri5' for adaptive RK, 'rk4' for fixed-step RK4)
            
        Returns:
            Latent trajectory of shape [num_steps+1, batch_size, target_size]
        """
        # Print integration method on first call
        if not hasattr(self, '_printed_ode_method'):
            if HAS_TORCHDIFFEQ:
                print(f"Using torchdiffeq with method '{method}' for ODE integration")
            else:
                print("Using manual RK4 for ODE integration (torchdiffeq not available)")
            self._printed_ode_method = True
        
        if HAS_TORCHDIFFEQ:
            # Use torchdiffeq for adaptive integration
            z_traj = odeint(
                self.koopman_ode_func,
                z0,
                t_span,
                method=method,
                rtol=1e-5,
                atol=1e-7,
            )
            return z_traj
        else:
            # Fallback: fixed-step RK4 (more accurate than Euler)
            return self._integrate_rk4_fallback(z0, t_span)
    
    def _integrate_rk4_fallback(
        self, 
        z0: torch.Tensor, 
        t_span: torch.Tensor
    ) -> torch.Tensor:
        """Fallback RK4 integration when torchdiffeq is not available.
        
        Implements classic 4th-order Runge-Kutta method.
        
        Args:
            z0: Initial latent state [batch_size, target_size]
            t_span: Time points [num_steps+1]
            
        Returns:
            Latent trajectory [num_steps+1, batch_size, target_size]
        """
        z_list = [z0]
        z = z0
        for i in range(len(t_span) - 1):
            t = t_span[i]
            dt = t_span[i+1] - t_span[i]
            
            # RK4 stages
            k1 = self.koopman_ode_func(t, z)
            k2 = self.koopman_ode_func(t + 0.5 * dt, z + 0.5 * dt * k1)
            k3 = self.koopman_ode_func(t + 0.5 * dt, z + 0.5 * dt * k2)
            k4 = self.koopman_ode_func(t + dt, z + dt * k3)
            
            # Update state
            z = z + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
            z_list.append(z)
        
        return torch.stack(z_list, dim=0)
    
    def rollout_sequence_ode(
        self,
        x0: torch.Tensor,
        num_steps: int,
        dt: float,
    ) -> torch.Tensor:
        """Rollout a sequence using ODE integration of Koopman dynamics.
        
        Args:
            x0: Initial observations of shape [batch_size, observation_size]
            num_steps: Number of steps to roll out
            dt: Time step between observations
            
        Returns:
            Predicted trajectory of shape [num_steps+1, batch_size, observation_size]
            Includes x0 at index 0
        """
        # Encode initial state
        z0 = self.encode(x0)  # [batch_size, target_size]
        
        # Create time span
        t_span = torch.arange(num_steps + 1, dtype=torch.float32, device=x0.device) * dt
        
        # Integrate latent dynamics
        z_traj = self.integrate_latent_ode(z0, t_span)  # [num_steps+1, batch_size, target_size]
        
        # Decode all latent states
        # Need to reshape for decoding: [num_steps+1 * batch_size, target_size]
        num_times, batch_size, target_size = z_traj.shape
        z_flat = z_traj.reshape(num_times * batch_size, target_size)
        x_flat = self.decode(z_flat)  # [num_times * batch_size, observation_size]
        x_traj = x_flat.reshape(num_times, batch_size, self.observation_size)
        
        return x_traj
    
    def loss(
        self,
        x: torch.Tensor,
        nx: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute total loss and metrics (single-step version, for backward compatibility).
        
        Args:
            x: Current states of shape [batch_size, observation_size]
            nx: Next states of shape [batch_size, observation_size]
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        # Linear prediction loss
        kmat = self.kmatrix()
        prediction = self.decode(self.encode(x) @ kmat)
        prediction_loss = torch.norm(prediction - nx, dim=-1).mean()
        
        # Linear dynamics alignment loss
        residual_loss = self.residual(x, nx).mean()
        
        # Reconstruction loss
        reconst_loss = torch.norm(x - self.reconstruction(x), dim=-1).mean()
        reconst_loss += torch.norm(nx - self.reconstruction(nx), dim=-1).mean()
        
        # Sparsity loss
        sparsity_loss = self.sparsity_loss(x)
        sparsity_loss += self.sparsity_loss(nx)
        sparsity_loss *= 0.5
        
        # Koopman matrix eigenvalues
        # MPS doesn't support eigvals, so move to CPU if needed
        with torch.no_grad():
            kmat_device = kmat.device
            if kmat_device.type == 'mps':
                kmat_cpu = kmat.cpu()
                eigvals = torch.linalg.eigvals(kmat_cpu)
            else:
                eigvals = torch.linalg.eigvals(kmat)
            max_eigenvalue = torch.max(eigvals.real)
        
        # Nonzero codes
        with torch.no_grad():
            z = self.encode(x)
            num_nonzero_codes = (z != 0).float().sum(dim=-1).mean()
            sparsity_ratio = 1.0 - num_nonzero_codes / self.target_size
        
        # Total weighted loss
        total_loss = (
            self.cfg.MODEL.RES_COEFF * residual_loss +
            self.cfg.MODEL.RECONST_COEFF * reconst_loss +
            self.cfg.MODEL.PRED_COEFF * prediction_loss +
            self.cfg.MODEL.SPARSITY_COEFF * sparsity_loss
        )
        
        metrics = {
            'loss': total_loss.item(),
            'residual_loss': residual_loss.item(),
            'reconst_loss': reconst_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'A_max_eigenvalue': max_eigenvalue.item(),
            'sparsity_ratio': sparsity_ratio.item(),
        }
        
        return total_loss, metrics
    
    def rollout_latent_discrete(
        self,
        z0: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Rollout discrete Koopman dynamics from initial latent state.
        
        Implements discrete dynamics: z_{t+k} = K^k z_0
        
        Args:
            z0: Initial latent state [batch_size, target_size]
            num_steps: Number of steps to roll forward (returns num_steps+1 states)
            
        Returns:
            Latent trajectory [batch_size, num_steps+1, target_size]
            Includes z0 at index 0, z1 at index 1, etc.
        """
        batch_size = z0.shape[0]
        kmat = self.kmatrix()
        
        # Collect trajectory
        z_list = [z0]
        z = z0
        for _ in range(num_steps):
            z = z @ kmat  # z_{t+1} = z_t @ K (discrete Koopman dynamics)
            z_list.append(z)
        
        # Stack: [num_steps+1, batch_size, target_size] -> [batch_size, num_steps+1, target_size]
        z_traj = torch.stack(z_list, dim=1)
        return z_traj
    
    def rollout_sequence(
        self,
        x0: torch.Tensor,
        num_steps: int,
    ) -> torch.Tensor:
        """Rollout discrete Koopman dynamics in observation space.
        
        Args:
            x0: Initial observations [batch_size, observation_size]
            num_steps: Number of steps to roll forward
            
        Returns:
            Predicted trajectory [batch_size, num_steps+1, observation_size]
            Includes x0 prediction at index 0
        """
        # Encode initial state
        z0 = self.encode(x0)  # [batch_size, target_size]
        
        # Rollout in latent space
        z_traj = self.rollout_latent_discrete(z0, num_steps)  # [batch_size, num_steps+1, target_size]
        
        # Decode all states
        batch_size, seq_len, target_size = z_traj.shape
        z_flat = z_traj.reshape(batch_size * seq_len, target_size)
        x_flat = self.decode(z_flat)
        x_traj = x_flat.reshape(batch_size, seq_len, self.observation_size)
        
        return x_traj

    def loss_sequence(
        self,
        x_seq: torch.Tensor,
        dt: float = 1.0,  # Unused for discrete dynamics, kept for API compatibility
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute sequence-based loss using discrete Koopman dynamics.
        
        This implements the discrete-time training scheme for finance:
        1. Encode all states in sequence: z_i = φ(x_i)
        2. Unroll discrete dynamics z_{t+k} = K^k z_t from z_0 to get predicted latents ẑ_i
        3. Decode both: x̃_i = ψ(z_i), x̂_i = ψ(ẑ_i)
        4. Compute alignment, reconstruction, and prediction losses
        
        This matches PDF Eq. 31: z_{t+1} = K z_t (discrete Koopman dynamics)
        
        Args:
            x_seq: Sequence of states with shape [batch_size, seq_len, observation_size]
                   Includes x_t, x_{t+1}, ..., x_{t+T}
            dt: Time step (unused for discrete dynamics, kept for API compatibility)
            
        Returns:
            Tuple of (total_loss, metrics_dict)
        """
        batch_size, seq_len, obs_size = x_seq.shape
        
        # 1. Encode each state in the sequence
        # Flatten for encoding: [batch_size * seq_len, obs_size]
        x_flat = x_seq.reshape(batch_size * seq_len, obs_size)
        z_flat = self.encode(x_flat)  # [batch_size * seq_len, target_size]
        z_seq = z_flat.reshape(batch_size, seq_len, self.target_size)
        
        # 2. Unroll discrete Koopman dynamics from initial state
        # z_{t+k} = K^k z_0 (discrete dynamics, PDF Eq. 31)
        z0 = z_seq[:, 0, :]  # [batch_size, target_size]
        
        # Rollout: z_hat_seq has shape [batch_size, seq_len, target_size]
        z_hat_seq = self.rollout_latent_discrete(z0, seq_len - 1)
        
        # 3. Decode both encoded and advanced latents
        # Reconstruction from encoded latents
        x_tilde = self.decode(z_flat).reshape(batch_size, seq_len, obs_size)
        
        # Prediction from discrete-advanced latents
        z_hat_flat = z_hat_seq.reshape(batch_size * seq_len, self.target_size)
        x_hat_flat = self.decode(z_hat_flat)
        x_hat_seq = x_hat_flat.reshape(batch_size, seq_len, obs_size)
        
        # 4. Compute losses
        
        # Alignment loss: sum over sequence (excluding initial state)
        # |ẑ_{t+i} - z_{t+i}|^2 for i = 1..T (PDF Eq. 34)
        alignment_loss = torch.norm(
            z_hat_seq[:, 1:, :] - z_seq[:, 1:, :], 
            dim=-1
        ).pow(2).sum(dim=1).mean()
        
        # Reconstruction loss: sum over entire sequence including initial
        # |x_{t+i} - x̃_{t+i}|^2 for i = 0..T (PDF Eq. 32)
        reconst_loss = torch.norm(
            x_seq - x_tilde,
            dim=-1
        ).pow(2).sum(dim=1).mean()
        
        # Prediction loss: sum over sequence (excluding initial state)
        # |x_{t+i} - x̂_{t+i}|^2 for i = 1..T (PDF Eq. 33)
        prediction_loss = torch.norm(
            x_seq[:, 1:, :] - x_hat_seq[:, 1:, :],
            dim=-1
        ).pow(2).sum(dim=1).mean()
        
        # Sparsity loss: L1 on latents averaged over sequence (PDF Eq. 35)
        sparsity_loss = torch.norm(z_seq, p=1, dim=-1).mean()
        
        # Metrics for monitoring
        with torch.no_grad():
            kmat = self.kmatrix()
            # MPS doesn't support eigvals, so move to CPU if needed
            kmat_device = kmat.device
            if kmat_device.type == 'mps':
                kmat_cpu = kmat.cpu()
                eigvals = torch.linalg.eigvals(kmat_cpu)
            else:
                eigvals = torch.linalg.eigvals(kmat)
            max_eigenvalue = torch.max(torch.abs(eigvals))
            
            num_nonzero_codes = (z_seq != 0).float().sum(dim=-1).mean()
            sparsity_ratio = 1.0 - num_nonzero_codes / self.target_size
        
        # Total weighted loss (PDF Eq. 36)
        total_loss = (
            self.cfg.MODEL.RES_COEFF * alignment_loss +
            self.cfg.MODEL.RECONST_COEFF * reconst_loss +
            self.cfg.MODEL.PRED_COEFF * prediction_loss +
            self.cfg.MODEL.SPARSITY_COEFF * sparsity_loss
        )
        
        metrics = {
            'loss': total_loss.item(),
            'residual_loss': alignment_loss.item(),  # Named 'residual' for consistency with single-step loss
            'reconst_loss': reconst_loss.item(),
            'prediction_loss': prediction_loss.item(),
            'sparsity_loss': sparsity_loss.item(),
            'A_max_eigenvalue': max_eigenvalue.item(),  # Named 'A' for consistency with single-step loss
            'sparsity_ratio': sparsity_ratio.item(),
        }
        
        return total_loss, metrics


# ---------------------------------------------------------------------------
# Concrete Implementations
# ---------------------------------------------------------------------------


class GenericKM(KoopmanMachine):
    """Generic Koopman Machine with MLP encoder and decoder.
    
    This is the standard Koopman autoencoder with configurable MLP architectures.
    Optionally supports normalization of latent codes.
    
    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space
    """
    
    def __init__(self, cfg: Config, observation_size: int):
        super().__init__(cfg, observation_size)
        
        # Encoder
        self.encoder = MLPCoder(
            input_size=observation_size,
            target_size=cfg.MODEL.TARGET_SIZE,
            hidden_layers=cfg.MODEL.ENCODER.LAYERS,
            use_bias=cfg.MODEL.ENCODER.USE_BIAS,
            last_relu=cfg.MODEL.ENCODER.LAST_RELU,
            activation=cfg.MODEL.ENCODER.ACTIVATION,
        )
        
        # Decoder
        self.decoder = MLPCoder(
            input_size=cfg.MODEL.TARGET_SIZE,
            target_size=observation_size,
            hidden_layers=cfg.MODEL.DECODER.LAYERS,
            use_bias=cfg.MODEL.DECODER.USE_BIAS,
            last_relu=False,
            activation=cfg.MODEL.DECODER.ACTIVATION,
        )
        
        # Koopman matrix (learnable)
        self.kmat = nn.Parameter(torch.eye(cfg.MODEL.TARGET_SIZE))

        self.norm_fn_name = cfg.MODEL.NORM_FN
    
    def _norm_fn(self, x: torch.Tensor) -> torch.Tensor:
        """Apply normalization to latent codes.
        
        Args:
            x: Latent codes
            
        Returns:
            Normalized latent codes
        """
        if self.norm_fn_name == 'id':
            return x
        elif self.norm_fn_name == 'ball':
            return x / torch.norm(x, dim=-1, keepdim=True)
        else:
            raise ValueError(f"Unknown norm function '{self.norm_fn_name}'")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations to latent space.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Latent codes of shape [..., target_size]
        """
        y = self.encoder(x)
        return self._norm_fn(y)
    
    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Decode latent codes to observation space.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Reconstructed observations of shape [..., observation_size]
        """
        return self.decoder(y)
    
    def kmatrix(self) -> torch.Tensor:
        """Get the Koopman matrix.
        
        Returns:
            Koopman matrix of shape [target_size, target_size]
        """
        return self.kmat
    
    def step_latent(self, y: torch.Tensor) -> torch.Tensor:
        """Step forward in latent space with normalization.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Next latent codes of shape [..., target_size]
        """
        ny = y @ self.kmatrix()
        return self._norm_fn(ny)

# TODO: test this class with experiments. Sweep over the sparsity coefficient values.
# TODO: test this on the Lyapunov environment
class LISTAKM(KoopmanMachine):
    """Koopman Machine with LISTA sparse encoder.
    
    Uses the Learned Iterative Soft-Thresholding Algorithm (LISTA) for sparse
    encoding. The decoder uses a normalized dictionary.
    
    Args:
        cfg: Configuration object
        observation_size: Dimension of the observation space
    """
    
    def __init__(self, cfg: Config, observation_size: int):
        super().__init__(cfg, observation_size)
        
        # Initialize dictionary
        # For LISTA, W_d is expected with shape [xdim, zdim].
        # The decoder dictionary parameter is stored as [zdim, xdim] for y @ W_d.
        Wd_init = torch.randn(observation_size, cfg.MODEL.TARGET_SIZE) * 0.01  # [xdim, zdim]
        self.register_buffer('dict_init', Wd_init.clone())
        self.dict = nn.Parameter(Wd_init.T)  # [zdim, xdim]
        
        # LISTA encoder
        self.lista = LISTA(cfg, observation_size, Wd_init)
        
        # Koopman matrix (learnable)
        self.kmat = nn.Parameter(torch.eye(cfg.MODEL.TARGET_SIZE))
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode observations using LISTA.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Sparse latent codes of shape [..., target_size]
        """
        return self.lista(x)
    
    def decode(self, y: torch.Tensor) -> torch.Tensor:
        """Decode using normalized dictionary.
        
        Args:
            y: Latent codes of shape [..., target_size]
            
        Returns:
            Reconstructed observations of shape [..., observation_size]
        """
        # Normalize dictionary atoms
        wd = self.dict / torch.norm(self.dict, dim=1, keepdim=True).clamp(min=1e-4)
        return y @ wd
    
    def kmatrix(self) -> torch.Tensor:
        """Get the Koopman matrix.
        
        Returns:
            Koopman matrix of shape [target_size, target_size]
        """
        return self.kmat
    
    def sparsity_loss(self, x: torch.Tensor) -> torch.Tensor:
        """Compute L1 sparsity loss weighted by LISTA alpha.
        
        Args:
            x: Observations of shape [..., observation_size]
            
        Returns:
            Scalar sparsity loss
        """
        z = self.encode(x)
        return self.cfg.MODEL.ENCODER.LISTA.ALPHA * torch.norm(z, p=1, dim=-1).mean()


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------


_MODEL_REGISTRY = {
    "GenericKM": GenericKM,
    "SparseKM": GenericKM,  # Same as GenericKM, configured via sparsity coeff
    "LISTAKM": LISTAKM,
}


def make_model(cfg: Config, observation_size: int) -> KoopmanMachine:
    """Factory function to create model from configuration.
    
    Args:
        cfg: Configuration object with MODEL.MODEL_NAME specifying the model type
        observation_size: Dimension of the observation space
        
    Returns:
        KoopmanMachine instance
        
    Raises:
        ValueError: If MODEL_NAME is not in registry
    """
    model_name = cfg.MODEL.MODEL_NAME
    if model_name not in _MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Available: {list(_MODEL_REGISTRY.keys())}"
        )
    return _MODEL_REGISTRY[model_name](cfg, observation_size)