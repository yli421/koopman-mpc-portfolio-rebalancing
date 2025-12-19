"""Configuration system using Python dataclasses for PyTorch implementation.

This module provides a type-safe, native Python configuration system using dataclasses.
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
    EMBEDDING_DIM: int = 10  # Number of lagged days in embedding
    CACHE_DIR: Optional[str] = ".cache/finance_data"  # Default cache directory
    SEQUENCE_LENGTH: int = 10  # >1 = sequence training (better for forecasting)
    RESAMPLE_WEEKLY: bool = False  # Resample data to weekly intervals


@dataclass
class EnvConfig:
    """Environment configuration."""
    ENV_NAME: str = "finance"
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
    NUM_WORKERS: int = 1  # number of dataloader workers
    
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
            ENV_NAME=env_dict.get("ENV_NAME", "finance"),
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
    """Create default configuration."""
    return Config()


def get_train_finance_sparse_config() -> Config:
    """Training configuration for finance portfolio rebalancing."""
    cfg = Config()
    cfg.ENV.ENV_NAME = "finance"
    
    # Model: GenericKM with sparsity - High-dimensional lifting strategy
    cfg.MODEL.MODEL_NAME = "GenericKM"
    cfg.MODEL.TARGET_SIZE = 1024  # Latent space dimension
    cfg.MODEL.NORM_FN = "id"
    
    # Encoder: MLP with gelu
    cfg.MODEL.ENCODER.LAYERS = [1024, 1024]
    cfg.MODEL.ENCODER.LAST_RELU = False
    cfg.MODEL.ENCODER.USE_BIAS = True
    cfg.MODEL.ENCODER.ACTIVATION = "gelu"  # GELU is smoother than ReLU and allows negative values
    
    # Decoder: Linear
    cfg.MODEL.DECODER.LAYERS = []
    cfg.MODEL.DECODER.USE_BIAS = False
    
    # Loss weights (tuned for finance)
    cfg.MODEL.RES_COEFF = 1.0
    cfg.MODEL.RECONST_COEFF = 0.02   
    cfg.MODEL.PRED_COEFF = 1.0     
    cfg.MODEL.SPARSITY_COEFF = 0.5
    
    # Training
    cfg.TRAIN.LR = 1e-3
    cfg.TRAIN.K_MATRIX_LR = 1e-4  # Slower and stabler learning for Koopman matrix
    cfg.TRAIN.NUM_STEPS = 10_000
    cfg.TRAIN.BATCH_SIZE = 256
    cfg.TRAIN.DATA_SIZE = 64 * 20
    cfg.TRAIN.USE_SEQUENCE_LOSS = True
    cfg.TRAIN.SEQUENCE_LENGTH = 10  # Sequence length for forecasting
    cfg.TRAIN.NUM_WORKERS = 1
    
    # Finance data config 
    # Enable data caching to avoid re-downloading
    cfg.ENV.FINANCE.CACHE_DIR = ".cache/finance_data"
    
    return cfg


_TRAIN_CONFIG_REGISTRY = {
    "finance_sparse": get_train_finance_sparse_config,
}


def get_config(name: str = "default") -> Config:
    """Get a named configuration."""
    if name == "default":
        return get_default_config()
    if name not in _TRAIN_CONFIG_REGISTRY:
        raise ValueError(f"Unknown config name '{name}'. Available: {list(_TRAIN_CONFIG_REGISTRY.keys())}")
    return _TRAIN_CONFIG_REGISTRY[name]()
