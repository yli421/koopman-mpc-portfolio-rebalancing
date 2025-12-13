"""Configuration system using Python dataclasses for PyTorch implementation.

This module provides a type-safe, native Python configuration system using dataclasses.
No external dependencies required (no ml_collections, no argparse conversion).

## Usage

### 1. Quick Start - Use Predefined Configs

```python
from config import get_config

# Get a named configuration
cfg = get_config("lista")  # Options: "generic", "generic_sparse", "lista", etc.

# Access nested fields with dot notation
print(cfg.MODEL.TARGET_SIZE)  # 2048
print(cfg.TRAIN.LR)  # 0.001
```

### 2. Modify Configs

```python
# Start with a base config and customize
cfg = get_config("generic")
cfg.MODEL.TARGET_SIZE = 128
cfg.TRAIN.BATCH_SIZE = 512
cfg.TRAIN.LR = 5e-4
cfg.ENV.ENV_NAME = "pendulum"
```

### 3. Create Custom Configs

```python
from config import Config, ModelConfig, TrainConfig

# Build from scratch
cfg = Config()
cfg.SEED = 42
cfg.MODEL.TARGET_SIZE = 256
cfg.TRAIN.NUM_STEPS = 5000

# Or use nested dataclasses
cfg = Config(
    SEED=42,
    MODEL=ModelConfig(TARGET_SIZE=256, SPARSITY_COEFF=0.01),
    TRAIN=TrainConfig(NUM_STEPS=5000, LR=1e-3)
)
```

### 4. Save and Load Configs

```python
# Save to JSON for reproducibility
cfg.to_json("experiments/run_001/config.json")

# Load from JSON
cfg = Config.from_json("experiments/run_001/config.json")

# Convert to dict for logging (e.g., wandb)
config_dict = cfg.to_dict()
```

### 5. Use in Training Code

```python
from config import get_config

cfg = get_config("lista")

# Pass config sections directly to your model/trainer
model = build_model(cfg.MODEL)
optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
dataloader = create_dataloader(cfg.ENV, cfg.TRAIN)

# Use loss coefficients
loss = (cfg.MODEL.RES_COEFF * alignment_loss + 
        cfg.MODEL.RECONST_COEFF * recon_loss +
        cfg.MODEL.SPARSITY_COEFF * sparsity_loss)
```

### 6. Notebook-Friendly

```python
# In Jupyter/IPython notebooks
cfg = get_config("lista")
cfg.TRAIN.NUM_STEPS = 1000  # Quick experiment
cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 5  # Faster iterations

# Configs are just Python objects - easy to inspect
cfg.MODEL  # Shows all model settings
```

## Available Configs

- `"default"`: Base configuration with minimal settings
- `"generic"`: Standard KoopmanAE with MLP encoder (64-dim latent)
- `"generic_sparse"`: KoopmanAE with L1 sparsity regularization
- `"generic_prediction"`: Prediction-focused (no reconstruction)
- `"lista"`: LISTA-based sparse autoencoder (2048-dim latent)
- `"lista_nonlinear"`: LISTA with nonlinear MLP encoder

## Structure

```
Config
├── SEED: int
├── ENV: EnvConfig
│   ├── ENV_NAME: str (system choice)
│   └── <SYSTEM>: SystemConfig (dt, params)
├── MODEL: ModelConfig
│   ├── MODEL_NAME: str
│   ├── TARGET_SIZE: int (latent dim)
│   ├── Loss coefficients (RES_COEFF, RECONST_COEFF, etc.)
│   ├── ENCODER: EncoderConfig
│   │   ├── LAYERS: List[int]
│   │   ├── ACTIVATION, USE_BIAS, etc.
│   │   └── LISTA: ListaConfig (for LISTAKM models)
│   └── DECODER: DecoderConfig
└── TRAIN: TrainConfig
    ├── NUM_STEPS: int (epochs)
    ├── BATCH_SIZE: int
    ├── LR: float
    └── DATA_SIZE: int
```
"""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import List, Optional
import json


# Default universe of liquid US stocks (diverse sectors)
DEFAULT_FINANCE_TICKERS = [
    # Technology
    "AAPL", "MSFT", "GOOGL", "AMZN", "META",
    # Finance
    "JPM", "BAC", "GS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "PFE", "ABBV",
    # Consumer
    "PG", "KO", "PEP", "WMT",
    # Energy & Industrials
    "XOM", "CVX",
]


@dataclass
class ParabolicConfig:
    """Parabolic attractor system parameters."""
    LAMBDA: float = -1.0
    MU: float = -0.1
    DT: float = 0.1


@dataclass
class DuffingConfig:
    """Duffing oscillator system parameters."""
    DT: float = 0.01


@dataclass
class PendulumConfig:
    """Pendulum system parameters."""
    DT: float = 0.01


@dataclass
class LotkaVolterraConfig:
    """Lotka-Volterra system parameters."""
    DT: float = 0.01


@dataclass
class Lorenz63Config:
    """Lorenz63 system parameters."""
    DT: float = 0.01


@dataclass
class LyapunovConfig:
    """Lyapunov multi-attractor system parameters."""
    DT: float = 0.05
    SIGMA: float = 0.5


@dataclass
class FinanceConfig:
    """Finance environment configuration for portfolio rebalancing.
    
    Attributes:
        TICKERS: List of stock tickers to use
        START_DATE: Start date for data download (YYYY-MM-DD)
        END_DATE: End date for data download (YYYY-MM-DD)
        TRAIN_END: End date for training split (YYYY-MM-DD)
        VAL_END: End date for validation split (YYYY-MM-DD)
        EMBEDDING_DIM: Time-delay embedding dimension (d in Y_t)
        CACHE_DIR: Directory to cache downloaded data (None = no caching)
        SEQUENCE_LENGTH: Sequence length for training (1 = pairwise)
    """
    TICKERS: List[str] = field(default_factory=lambda: DEFAULT_FINANCE_TICKERS.copy())
    START_DATE: str = "2012-01-01"
    END_DATE: str = "2024-12-31"
    TRAIN_END: str = "2018-12-31"
    VAL_END: str = "2020-12-31"
    EMBEDDING_DIM: int = 20  # Number of lagged days in embedding
    CACHE_DIR: Optional[str] = None
    SEQUENCE_LENGTH: int = 10  # >1 = sequence training (better for forecasting)


@dataclass
class EnvConfig:
    """Environment configuration."""
    ENV_NAME: str = "duffing"  # from ["duffing", "parabolic", "pendulum", "lotka_volterra", "lorenz63", "finance"]
    PARABOLIC: ParabolicConfig = field(default_factory=ParabolicConfig)
    DUFFING: DuffingConfig = field(default_factory=DuffingConfig)
    PENDULUM: PendulumConfig = field(default_factory=PendulumConfig)
    LOTKA_VOLTERRA: LotkaVolterraConfig = field(default_factory=LotkaVolterraConfig)
    LORENZ63: Lorenz63Config = field(default_factory=Lorenz63Config)
    LYAPUNOV: LyapunovConfig = field(default_factory=LyapunovConfig)
    FINANCE: FinanceConfig = field(default_factory=FinanceConfig)


@dataclass
class ListaConfig:
    """LISTA encoder-specific configuration."""
    NUM_LOOPS: int = 10  # LISTA iterations
    L: float = 1e3  # Lipschitz constant estimate
    ALPHA: float = 0.1  # sparsity threshold
    LINEAR_ENCODER: bool = False  # use MLP vs linear encoder


@dataclass
class EncoderConfig:
    """Encoder architecture configuration."""
    LAYERS: List[int] = field(default_factory=lambda: [16, 16])  # hidden layer sizes
    LAST_RELU: bool = False
    USE_BIAS: bool = False
    ACTIVATION: str = "relu"  # from ["relu", "tanh", "gelu"]
    LISTA: ListaConfig = field(default_factory=ListaConfig)


@dataclass
class DecoderConfig:
    """Decoder architecture configuration."""
    LAYERS: List[int] = field(default_factory=list)  # linear decoder by default
    USE_BIAS: bool = False
    ACTIVATION: str = "relu"


@dataclass
class ModelConfig:
    """Model architecture and loss configuration."""
    MODEL_NAME: str = "SparseKM"  # from ["GenericKM", "SparseKM", "LISTAKM"]
    NORM_FN: str = "id"  # from ["id", "ball"]
    TARGET_SIZE: int = 16  # latent_dim i.e. zdim
    
    # Loss coefficients
    RES_COEFF: float = 1.0  # alignment loss weight
    RECONST_COEFF: float = 0.02  # reconstruction loss weight
    PRED_COEFF: float = 0.0  # prediction loss weight
    SPARSITY_COEFF: float = 1e-3  # sparsity loss weight (L1 regularization)
    
    # Sub-configs
    ENCODER: EncoderConfig = field(default_factory=EncoderConfig)
    DECODER: DecoderConfig = field(default_factory=DecoderConfig)


@dataclass
class TrainConfig:
    """Training configuration."""
    NUM_STEPS: int = 2_000  # total training steps (epochs)
    BATCH_SIZE: int = 256
    DATA_SIZE: int = 256 * 8  # total dataset size
    LR: float = 1e-4  # main learning rate (encoder/decoder)
    WEIGHT_DECAY: float = 1e-4  # weight decay for AdamW optimizer
    K_MATRIX_LR: float = 1e-5  # learning rate for Koopman matrix parameters
    
    # Sequence training parameters
    USE_SEQUENCE_LOSS: bool = False  # default to single-step loss for parity with JAX
    SEQUENCE_LENGTH: int = 10  # number of forward steps in each training sequence (T)

@dataclass
class Config:
    """Main configuration container."""
    SEED: int = 0
    ENV: EnvConfig = field(default_factory=EnvConfig)
    MODEL: ModelConfig = field(default_factory=ModelConfig)
    TRAIN: TrainConfig = field(default_factory=TrainConfig)
    
    def to_dict(self) -> dict:
        """Convert config to dictionary."""
        return asdict(self)
    
    def to_json(self, filepath: str) -> None:
        """Save config to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> Config:
        """Create config from dictionary."""
        # Recursively construct nested dataclasses
        env_dict = config_dict.get("ENV", {})
        env = EnvConfig(
            ENV_NAME=env_dict.get("ENV_NAME", "duffing"),
            PARABOLIC=ParabolicConfig(**env_dict.get("PARABOLIC", {})),
            DUFFING=DuffingConfig(**env_dict.get("DUFFING", {})),
            PENDULUM=PendulumConfig(**env_dict.get("PENDULUM", {})),
            LOTKA_VOLTERRA=LotkaVolterraConfig(**env_dict.get("LOTKA_VOLTERRA", {})),
            LORENZ63=Lorenz63Config(**env_dict.get("LORENZ63", {})),
            LYAPUNOV=LyapunovConfig(**env_dict.get("LYAPUNOV", {})),
            FINANCE=FinanceConfig(**env_dict.get("FINANCE", {})),
        )
        
        model_dict = config_dict.get("MODEL", {})
        encoder_dict = model_dict.get("ENCODER", {})
        lista = ListaConfig(**encoder_dict.get("LISTA", {}))
        encoder = EncoderConfig(**{k: v for k, v in encoder_dict.items() if k != "LISTA"})
        encoder.LISTA = lista
        decoder = DecoderConfig(**model_dict.get("DECODER", {}))
        
        model = ModelConfig(**{k: v for k, v in model_dict.items() if k not in ["ENCODER", "DECODER"]})
        model.ENCODER = encoder
        model.DECODER = decoder
        
        train = TrainConfig(**config_dict.get("TRAIN", {}))
        
        return cls(
            SEED=config_dict.get("SEED", 0),
            ENV=env,
            MODEL=model,
            TRAIN=train
        )
    
    @classmethod
    def from_json(cls, filepath: str) -> Config:
        """Load config from JSON file."""
        with open(filepath, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)


def get_default_config() -> Config:
    """Create default configuration.
    
    Returns:
        Config with default settings.
    """
    return Config()


def get_train_generic_km_config() -> Config:
    """Training configuration for GenericKM (standard Koopman AE with MLP encoder)."""
    cfg = Config()
    cfg.TRAIN.LR = 1e-4
    cfg.MODEL.MODEL_NAME = "GenericKM"
    cfg.MODEL.TARGET_SIZE = 64
    cfg.MODEL.NORM_FN = "id"
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.ENCODER.LAYERS = [64, 64]
    cfg.MODEL.SPARSITY_COEFF = 0.0
    return cfg


def get_train_generic_sparse_config() -> Config:
    """Training configuration for GenericKM with L1 regularization."""
    cfg = Config()
    cfg.TRAIN.LR = 1e-4
    cfg.MODEL.MODEL_NAME = "GenericKM"
    cfg.MODEL.TARGET_SIZE = 64
    cfg.MODEL.NORM_FN = "id"
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.ENCODER.LAYERS = [64, 64]
    cfg.MODEL.ENCODER.LAST_RELU = True
    cfg.MODEL.ENCODER.USE_BIAS = True
    cfg.MODEL.RECONST_COEFF = 0.5
    cfg.MODEL.SPARSITY_COEFF = 0.01
    return cfg


def get_train_generic_prediction_config() -> Config:
    """Training configuration for prediction-focused KoopmanAE."""
    cfg = Config()
    cfg.MODEL.MODEL_NAME = "GenericKM"
    cfg.TRAIN.LR = 1e-3
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.PRED_COEFF = 1.0
    cfg.MODEL.RES_COEFF = 0.0
    cfg.MODEL.RECONST_COEFF = 0.0
    cfg.MODEL.SPARSITY_COEFF = 0.0
    return cfg


def get_train_lista_config() -> Config:
    """Configuration for LISTA-based Sparse KM."""
    cfg = Config()
    cfg.MODEL.MODEL_NAME = "LISTAKM"
    cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = True
    cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 10
    cfg.MODEL.TARGET_SIZE = 1024 * 2
    cfg.MODEL.RES_COEFF = 1.0
    cfg.MODEL.RECONST_COEFF = 1.0
    cfg.MODEL.PRED_COEFF = 0.0
    cfg.MODEL.SPARSITY_COEFF = 1.0
    cfg.MODEL.NORM_FN = "id"
    cfg.MODEL.ENCODER.LISTA.L = 0.1
    cfg.MODEL.ENCODER.LISTA.ALPHA = 5e-3
    return cfg


def get_train_lista_nonlinear_config() -> Config:
    """Training configuration for LISTA with nonlinear encoder."""
    cfg = Config()
    cfg.MODEL.MODEL_NAME = "LISTAKM"
    cfg.MODEL.ENCODER.LISTA.LINEAR_ENCODER = False
    cfg.MODEL.ENCODER.LAYERS = [64, 64, 64]
    cfg.MODEL.ENCODER.LISTA.NUM_LOOPS = 10
    cfg.MODEL.TARGET_SIZE = 1024 * 2
    cfg.MODEL.RES_COEFF = 1.0
    cfg.MODEL.RECONST_COEFF = 1.0
    cfg.MODEL.PRED_COEFF = 0.0
    cfg.MODEL.SPARSITY_COEFF = 1.0
    cfg.MODEL.NORM_FN = "id"
    cfg.MODEL.ENCODER.LISTA.L = 1e4
    cfg.MODEL.ENCODER.LISTA.ALPHA = 1.0
    cfg.MODEL.ENCODER.LAST_RELU = True
    cfg.MODEL.ENCODER.USE_BIAS = True
    return cfg


def get_train_finance_sparse_config() -> Config:
    """Training configuration for finance portfolio rebalancing.
    
    This config is designed for learning Koopman representations of 
    financial market dynamics using log-returns with time-delay embedding.
    
    Key design choices:
    - Pairwise training (USE_SEQUENCE_LOSS=False) for initial experiments
    - Moderate sparsity regularization
    - ReLU encoder with bias for expressiveness
    - Linear decoder for simplicity
    """
    cfg = Config()
    cfg.ENV.ENV_NAME = "finance"
    
    # Model: GenericKM with sparsity
    cfg.MODEL.MODEL_NAME = "GenericKM"
    cfg.MODEL.TARGET_SIZE = 128  # Latent dimension (can increase later)
    cfg.MODEL.NORM_FN = "id"
    
    # Encoder: MLP with ReLU
    cfg.MODEL.ENCODER.LAYERS = [128, 128]
    cfg.MODEL.ENCODER.LAST_RELU = False  # Allow negative latents to avoid collapse
    cfg.MODEL.ENCODER.USE_BIAS = True
    cfg.MODEL.ENCODER.ACTIVATION = "relu"
    
    # Decoder: Linear
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.DECODER.USE_BIAS = False
    
    # Loss weights (tuned for finance)
    cfg.MODEL.RES_COEFF = 1.0      # Lower alignment to prevent zero-collapse
    cfg.MODEL.RECONST_COEFF = 0.02  # High recon to force learning features
    cfg.MODEL.PRED_COEFF = 1.0     # Prediction loss (important for forecasting)
    cfg.MODEL.SPARSITY_COEFF = 0.0  # Disable sparsity initially to prevent collapse
    
    # Training
    cfg.TRAIN.LR = 1e-4
    cfg.TRAIN.K_MATRIX_LR = 1e-5  # Slower learning for Koopman matrix
    cfg.TRAIN.NUM_STEPS = 10_000
    cfg.TRAIN.BATCH_SIZE = 64  # Smaller batches for finance (less data)
    cfg.TRAIN.DATA_SIZE = 64 * 20
    cfg.TRAIN.USE_SEQUENCE_LOSS = True  # Use sequence loss for multi-step stability
    cfg.TRAIN.SEQUENCE_LENGTH = 10      # Matches data config
    
    # Finance data config 
    # Enable data caching to avoid re-downloading
    cfg.ENV.FINANCE.CACHE_DIR = ".cache/finance_data"
    
    return cfg


_TRAIN_CONFIG_REGISTRY = {
    "generic": get_train_generic_km_config,
    "generic_sparse": get_train_generic_sparse_config,
    "generic_prediction": get_train_generic_prediction_config,
    "lista": get_train_lista_config,
    "lista_nonlinear": get_train_lista_nonlinear_config,
    "finance_sparse": get_train_finance_sparse_config,
}


def get_config(name: str = "default") -> Config:
    """Get a named configuration.
    
    Args:
        name: Configuration name. Options:
            - "default": Base configuration
            - "generic": Standard KoopmanMachine
            - "generic_sparse": Sparse KoopmanMachine with L1
            - "generic_prediction": Prediction-focused
            - "lista": LISTA-based KoopmanMachine
            - "lista_nonlinear": LISTA with MLP encoder
            - "finance_sparse": Finance portfolio rebalancing
    
    Returns:
        Config for the specified configuration.
    """
    if name == "default":
        return get_default_config()
    if name not in _TRAIN_CONFIG_REGISTRY:
        raise ValueError(f"Unknown config name '{name}'. Available: {list(_TRAIN_CONFIG_REGISTRY.keys())}")
    return _TRAIN_CONFIG_REGISTRY[name]()

