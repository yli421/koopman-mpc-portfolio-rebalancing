"""
Finance data module for Koopman-MPC portfolio rebalancing.

This module provides:
- Stock data downloading via yfinance API
- Log-return computation and standardization
- Time-delay embedding construction
- PyTorch Dataset for pairwise and sequence training
- Chronological train/val/test splitting

Based on the project PDF (IFT6162):
- Observations: Y_t = [y_t, y_{t-1}, ..., y_{t-d+1}] (time-delay embedding)
- Returns: y_t = log(p_t) - log(p_{t-1})
- Chronological splits: Train 2012-2018, Val 2018-2020, Test 2021-2024
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, List, Dict, Union
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# Default universe of liquid US stocks (diverse sectors)
DEFAULT_TICKERS = [
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


# ---------------------------------------------------------------------------
# Data Classes for Configuration
# ---------------------------------------------------------------------------


@dataclass
class FinanceDataConfig:
    """Configuration for finance data loading and processing.
    
    Attributes:
        tickers: List of stock tickers to download
        start_date: Start date for data download (YYYY-MM-DD)
        end_date: End date for data download (YYYY-MM-DD)
        train_end: End date for training split (YYYY-MM-DD)
        val_end: End date for validation split (YYYY-MM-DD)
        embedding_dim: Dimension of time-delay embedding (d in Y_t)
        cache_dir: Directory to cache downloaded data
    """
    tickers: List[str] = field(default_factory=lambda: DEFAULT_TICKERS.copy())
    start_date: str = "2012-01-01"
    end_date: str = "2024-12-31"
    train_end: str = "2018-12-31"
    val_end: str = "2020-12-31"
    embedding_dim: int = 5  # Number of lagged days in embedding
    cache_dir: Optional[str] = None


@dataclass 
class FinanceStats:
    """Statistics for standardization (from training data only).
    
    Attributes:
        mean: Mean of log-returns per asset [n_assets]
        std: Standard deviation of log-returns per asset [n_assets]
        tickers: List of ticker symbols (for reference)
    """
    mean: np.ndarray
    std: np.ndarray
    tickers: List[str]


# ---------------------------------------------------------------------------
# Core Data Processing Functions
# ---------------------------------------------------------------------------


def download_stock_data(
    tickers: List[str],
    start_date: str,
    end_date: str,
    cache_path: Optional[Path] = None,
) -> pd.DataFrame:
    """Download adjusted close prices from Yahoo Finance.
    
    Args:
        tickers: List of stock ticker symbols
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        cache_path: Optional path to save/load cached data
        
    Returns:
        DataFrame with dates as index and tickers as columns,
        containing adjusted close prices.
        
    Note:
        Uses yfinance API. Network access required unless cached.
    """
    import yfinance as yf
    
    # Check cache first
    if cache_path is not None and cache_path.exists():
        print(f"Loading cached data from {cache_path}")
        return pd.read_parquet(cache_path)
    
    print(f"Downloading {len(tickers)} tickers from {start_date} to {end_date}...")
    
    # Download data
    data = yf.download(
        tickers, 
        start=start_date, 
        end=end_date,
        auto_adjust=True,  # Use adjusted close
        progress=True
    )
    
    # Handle single ticker case (yf returns different structure)
    if len(tickers) == 1:
        prices = data['Close'].to_frame(name=tickers[0])
    else:
        prices = data['Close']
    
    # Ensure columns are tickers
    prices.columns = [str(col) for col in prices.columns]
    
    # Cache if path provided
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_parquet(cache_path)
        print(f"Cached data to {cache_path}")
    
    return prices


def clean_price_data(
    prices: pd.DataFrame,
    max_missing_ratio: float = 0.1,
    max_gap_days: int = 5,
) -> pd.DataFrame:
    """Clean and align price data.
    
    Operations:
    1. Drop assets with too much missing data
    2. Forward-fill short gaps (up to max_gap_days)
    3. Drop remaining rows with any NaN
    
    Args:
        prices: DataFrame of adjusted close prices
        max_missing_ratio: Maximum ratio of missing values per asset to keep
        max_gap_days: Maximum gap size to forward-fill
        
    Returns:
        Cleaned DataFrame with no missing values
    """
    n_original = len(prices)
    n_assets_original = len(prices.columns)
    
    # Calculate missing ratio per asset
    missing_ratios = prices.isna().mean()
    
    # Keep only assets with acceptable missing ratio
    good_assets = missing_ratios[missing_ratios <= max_missing_ratio].index
    prices = prices[good_assets].copy()
    
    n_assets_kept = len(prices.columns)
    if n_assets_kept < n_assets_original:
        dropped = set(missing_ratios.index) - set(good_assets)
        print(f"Dropped {n_assets_original - n_assets_kept} assets with >={max_missing_ratio*100:.0f}% missing: {dropped}")
    
    # Forward fill short gaps (limit to max_gap_days)
    prices = prices.ffill(limit=max_gap_days)
    
    # Drop any remaining rows with NaN
    prices = prices.dropna()
    
    n_final = len(prices)
    if n_final < n_original:
        print(f"Dropped {n_original - n_final} rows with missing values (remaining: {n_final})")
    
    return prices


def compute_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute log-returns from prices.
    
    Formula: y_t = log(p_t) - log(p_{t-1}) = log(p_t / p_{t-1})
    
    Args:
        prices: DataFrame of adjusted close prices
        
    Returns:
        DataFrame of log-returns (first row is NaN and dropped)
    """
    log_prices = np.log(prices)
    log_returns = log_prices.diff().iloc[1:]  # Drop first NaN row
    return log_returns


def compute_standardization_stats(
    log_returns: pd.DataFrame,
    train_end: str,
) -> FinanceStats:
    """Compute standardization statistics from training data only.
    
    Args:
        log_returns: DataFrame of log-returns
        train_end: End date for training period (YYYY-MM-DD)
        
    Returns:
        FinanceStats with mean and std computed from training data
    """
    # Get training data only
    train_data = log_returns[log_returns.index <= train_end]
    
    if len(train_data) == 0:
        raise ValueError(f"No training data before {train_end}")
    
    mean = train_data.mean().values
    std = train_data.std().values
    
    # Avoid division by zero
    std = np.maximum(std, 1e-8)
    
    return FinanceStats(
        mean=mean,
        std=std,
        tickers=list(log_returns.columns)
    )


def standardize_returns(
    log_returns: pd.DataFrame,
    stats: FinanceStats,
) -> pd.DataFrame:
    """Standardize log-returns using pre-computed statistics.
    
    Formula: z = (y - mean) / std
    
    Args:
        log_returns: DataFrame of log-returns
        stats: Pre-computed statistics from training data
        
    Returns:
        Standardized log-returns DataFrame
    """
    standardized = (log_returns - stats.mean) / stats.std
    return standardized


def time_delay_embedding(
    data: np.ndarray,
    embedding_dim: int,
) -> np.ndarray:
    """Create time-delay embedded observations.
    
    For each time t, creates Y_t = [y_t, y_{t-1}, ..., y_{t-d+1}]
    where d = embedding_dim.
    
    Args:
        data: Array of shape [T, n_assets] (time series of returns)
        embedding_dim: Number of lagged observations to include
        
    Returns:
        Embedded array of shape [T - d + 1, d * n_assets]
        Each row is a flattened [y_t; y_{t-1}; ...; y_{t-d+1}]
    """
    T, n_assets = data.shape
    
    if T < embedding_dim:
        raise ValueError(f"Time series length {T} < embedding_dim {embedding_dim}")
    
    # Number of valid embedded observations
    n_embedded = T - embedding_dim + 1
    
    # Pre-allocate output
    embedded = np.zeros((n_embedded, embedding_dim * n_assets), dtype=data.dtype)
    
    for i in range(n_embedded):
        # Stack [y_t, y_{t-1}, ..., y_{t-d+1}] as a flat vector
        # y_t is at position i + embedding_dim - 1, y_{t-d+1} is at position i
        for j in range(embedding_dim):
            start_col = j * n_assets
            end_col = (j + 1) * n_assets
            # j=0 corresponds to y_t (most recent)
            # j=embedding_dim-1 corresponds to y_{t-d+1} (oldest)
            embedded[i, start_col:end_col] = data[i + embedding_dim - 1 - j]
    
    return embedded


def create_finance_splits(
    log_returns: pd.DataFrame,
    stats: FinanceStats,
    train_end: str,
    val_end: str,
    embedding_dim: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Create chronological train/val/test splits with time-delay embedding.
    
    Splits are leak-free: standardization uses only training statistics,
    and temporal boundaries respect the embedding window.
    
    Args:
        log_returns: DataFrame of log-returns
        stats: Standardization statistics from training data
        train_end: End date for training (YYYY-MM-DD)
        val_end: End date for validation (YYYY-MM-DD)
        embedding_dim: Time-delay embedding dimension
        
    Returns:
        Tuple of (train_data, train_dates, val_data, val_dates, test_data, test_dates)
        Each data array has shape [n_samples, embedding_dim * n_assets]
        Each dates array has the corresponding date indices
    """
    # Standardize all data using training statistics
    standardized = standardize_returns(log_returns, stats)
    
    # Convert to numpy
    data = standardized.values.astype(np.float32)
    dates = standardized.index
    
    # Create embedding
    embedded = time_delay_embedding(data, embedding_dim)
    
    # Adjust dates to account for embedding (first embedding_dim-1 dates are lost)
    embedded_dates = dates[embedding_dim - 1:]
    
    # Split by date
    train_mask = embedded_dates <= train_end
    val_mask = (embedded_dates > train_end) & (embedded_dates <= val_end)
    test_mask = embedded_dates > val_end
    
    train_data = embedded[train_mask]
    val_data = embedded[val_mask]
    test_data = embedded[test_mask]
    
    train_dates = embedded_dates[train_mask]
    val_dates = embedded_dates[val_mask]
    test_dates = embedded_dates[test_mask]
    
    return train_data, train_dates, val_data, val_dates, test_data, test_dates


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------


class FinanceDataset(Dataset):
    """PyTorch Dataset for finance time series.
    
    Yields either:
    - Pairwise samples (Y_t, Y_{t+1}) for single-step training
    - Sequence samples (Y_t, Y_{t+1}, ..., Y_{t+T}) for multi-step training
    
    The dataset stores embedded observations where each Y_t is the
    time-delay embedding [y_t, y_{t-1}, ..., y_{t-d+1}].
    
    Args:
        data: Embedded observations of shape [n_samples, embedding_size]
        dates: Corresponding date indices (optional, for debugging)
        sequence_length: If >1, return sequences of this length; if 1, return pairs
    """
    
    def __init__(
        self,
        data: np.ndarray,
        dates: Optional[pd.DatetimeIndex] = None,
        sequence_length: int = 1,
    ):
        self.data = torch.from_numpy(data).float()
        self.dates = dates
        self.sequence_length = sequence_length
        
        # For pairwise: each sample is (Y_t, Y_{t+1}), so we lose 1 sample
        # For sequence: we lose sequence_length samples
        self.n_samples = len(data) - sequence_length
        
        if self.n_samples <= 0:
            raise ValueError(
                f"Data length {len(data)} too short for sequence_length {sequence_length}"
            )
    
    def __len__(self) -> int:
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Union[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        """Get a sample.
        
        Args:
            idx: Sample index
            
        Returns:
            If sequence_length == 1: Tuple (Y_t, Y_{t+1})
            If sequence_length > 1: Tensor of shape [sequence_length+1, embedding_size]
        """
        if self.sequence_length == 1:
            # Pairwise: return (Y_t, Y_{t+1})
            return self.data[idx], self.data[idx + 1]
        else:
            # Sequence: return [Y_t, Y_{t+1}, ..., Y_{t+T}]
            return self.data[idx:idx + self.sequence_length + 1]
    
    @property
    def observation_size(self) -> int:
        """Size of each observation (embedding_dim * n_assets)."""
        return self.data.shape[1]


# ---------------------------------------------------------------------------
# High-Level Data Loading Function
# ---------------------------------------------------------------------------


def load_finance_data(
    config: Optional[FinanceDataConfig] = None,
    sequence_length: int = 1,
) -> Tuple[FinanceDataset, FinanceDataset, FinanceDataset, FinanceStats, Dict]:
    """Load and prepare finance data for training.
    
    This is the main entry point for getting finance data ready for Koopman training.
    
    Args:
        config: Finance data configuration (uses defaults if None)
        sequence_length: Sequence length for dataset (1 for pairwise)
        
    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset, stats, metadata)
        metadata contains info about tickers, dates, shapes, etc.
    """
    if config is None:
        config = FinanceDataConfig()
    
    # Set up cache path
    cache_path = None
    if config.cache_dir is not None:
        cache_dir = Path(config.cache_dir)
        # Create unique cache filename based on tickers and dates
        ticker_hash = hash(tuple(sorted(config.tickers))) % 10000
        cache_path = cache_dir / f"prices_{config.start_date}_{config.end_date}_{ticker_hash}.parquet"
    
    # Download and clean data
    prices = download_stock_data(
        tickers=config.tickers,
        start_date=config.start_date,
        end_date=config.end_date,
        cache_path=cache_path,
    )
    prices = clean_price_data(prices)
    
    # Compute log-returns
    log_returns = compute_log_returns(prices)
    
    # Compute standardization stats from training data only
    stats = compute_standardization_stats(log_returns, config.train_end)
    
    # Create splits with embedding
    train_data, train_dates, val_data, val_dates, test_data, test_dates = create_finance_splits(
        log_returns=log_returns,
        stats=stats,
        train_end=config.train_end,
        val_end=config.val_end,
        embedding_dim=config.embedding_dim,
    )
    
    # Create datasets
    train_dataset = FinanceDataset(train_data, train_dates, sequence_length)
    val_dataset = FinanceDataset(val_data, val_dates, sequence_length)
    test_dataset = FinanceDataset(test_data, test_dates, sequence_length)
    
    # Collect metadata
    metadata = {
        "tickers": list(log_returns.columns),
        "n_assets": len(log_returns.columns),
        "embedding_dim": config.embedding_dim,
        "observation_size": train_dataset.observation_size,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "test_samples": len(test_dataset),
        "train_date_range": (str(train_dates[0].date()), str(train_dates[-1].date())),
        "val_date_range": (str(val_dates[0].date()), str(val_dates[-1].date())),
        "test_date_range": (str(test_dates[0].date()), str(test_dates[-1].date())),
        "prices_shape": prices.shape,
        "log_returns_shape": log_returns.shape,
    }
    
    print(f"\nFinance data loaded:")
    print(f"  Tickers: {len(metadata['tickers'])} assets")
    print(f"  Embedding dim: {metadata['embedding_dim']}")
    print(f"  Observation size: {metadata['observation_size']}")
    print(f"  Train: {metadata['train_samples']} samples ({metadata['train_date_range'][0]} to {metadata['train_date_range'][1]})")
    print(f"  Val: {metadata['val_samples']} samples ({metadata['val_date_range'][0]} to {metadata['val_date_range'][1]})")
    print(f"  Test: {metadata['test_samples']} samples ({metadata['test_date_range'][0]} to {metadata['test_date_range'][1]})")
    
    return train_dataset, val_dataset, test_dataset, stats, metadata


# ---------------------------------------------------------------------------
# Utility Functions for Analysis
# ---------------------------------------------------------------------------


def verify_embedding_shift(embedded: np.ndarray, n_assets: int, embedding_dim: int) -> bool:
    """Verify that Y_{t+1} is a shifted version of Y_t plus new y_{t+1}.
    
    The embedding Y_t = [y_t, y_{t-1}, ..., y_{t-d+1}] should satisfy:
    Y_{t+1}[1:d] == Y_t[0:d-1] (the overlap)
    
    Args:
        embedded: Embedded observations [n_samples, embedding_dim * n_assets]
        n_assets: Number of assets
        embedding_dim: Embedding dimension d
        
    Returns:
        True if the shift property holds for all consecutive pairs
    """
    for i in range(len(embedded) - 1):
        Y_t = embedded[i].reshape(embedding_dim, n_assets)
        Y_t1 = embedded[i + 1].reshape(embedding_dim, n_assets)
        
        # Y_{t+1}[1:] should equal Y_t[:-1]
        # Because Y_t = [y_t, y_{t-1}, ..., y_{t-d+1}]
        # And Y_{t+1} = [y_{t+1}, y_t, ..., y_{t-d+2}]
        # So Y_{t+1}[1:] = [y_t, ..., y_{t-d+2}] = Y_t[:-1]
        if not np.allclose(Y_t1[1:], Y_t[:-1]):
            return False
    
    return True


def compute_return_stats(log_returns: pd.DataFrame) -> pd.DataFrame:
    """Compute summary statistics for log-returns.
    
    Args:
        log_returns: DataFrame of log-returns
        
    Returns:
        DataFrame with statistics per asset
    """
    stats = pd.DataFrame({
        'mean': log_returns.mean(),
        'std': log_returns.std(),
        'min': log_returns.min(),
        'max': log_returns.max(),
        'skew': log_returns.skew(),
        'kurtosis': log_returns.kurtosis(),
        'missing_ratio': log_returns.isna().mean(),
    })
    return stats


def compute_autocorrelation(log_returns: pd.DataFrame, lag: int = 1) -> pd.Series:
    """Compute autocorrelation of log-returns at specified lag.
    
    Args:
        log_returns: DataFrame of log-returns
        lag: Lag for autocorrelation
        
    Returns:
        Series of autocorrelation values per asset
    """
    return log_returns.apply(lambda x: x.autocorr(lag=lag))


# ---------------------------------------------------------------------------
# Finance Environment Wrapper for Training Integration
# ---------------------------------------------------------------------------


class FinanceEnv:
    """Environment-like wrapper for finance data to integrate with train.py.
    
    This class provides an interface similar to the dynamical system environments
    in data.py, but works with pre-recorded financial data instead of simulating
    dynamics. This allows the finance data to be used with the existing training
    infrastructure.
    
    Key difference from dynamical systems:
    - For dynamical systems: step(x_t) -> x_{t+1} via ODE integration
    - For finance: step(x_t, idx) -> x_{t+1} from recorded data
    
    Since we can't "step" finance data arbitrarily, this wrapper is primarily
    used for:
    1. Providing observation_size for model construction
    2. Providing DataLoader for training
    3. Providing test sequences for evaluation
    
    Args:
        train_dataset: FinanceDataset for training
        val_dataset: FinanceDataset for validation
        test_dataset: FinanceDataset for testing
        stats: Standardization statistics
        metadata: Dataset metadata
    """
    
    def __init__(
        self,
        train_dataset: FinanceDataset,
        val_dataset: FinanceDataset,
        test_dataset: FinanceDataset,
        stats: FinanceStats,
        metadata: Dict,
    ):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.stats = stats
        self.metadata = metadata
        self._observation_size = train_dataset.observation_size
    
    @property
    def observation_size(self) -> int:
        """Dimension of each observation (embedding_dim * n_assets)."""
        return self._observation_size
    
    @property
    def n_assets(self) -> int:
        """Number of assets in the universe."""
        return self.metadata['n_assets']
    
    @property
    def embedding_dim(self) -> int:
        """Time-delay embedding dimension."""
        return self.metadata['embedding_dim']
    
    def get_dataloader(
        self,
        split: str = 'train',
        batch_size: int = 64,
        shuffle: bool = True,
    ) -> torch.utils.data.DataLoader:
        """Get a PyTorch DataLoader for the specified split.
        
        Args:
            split: One of 'train', 'val', 'test'
            batch_size: Batch size for the DataLoader
            shuffle: Whether to shuffle the data
            
        Returns:
            DataLoader for the specified split
        """
        from torch.utils.data import DataLoader
        
        if split == 'train':
            dataset = self.train_dataset
        elif split == 'val':
            dataset = self.val_dataset
        elif split == 'test':
            dataset = self.test_dataset
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train', 'val', or 'test'.")
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            drop_last=True,
        )
    
    def get_test_sequences(
        self,
        num_sequences: int = 100,
        max_length: int = 200,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get test sequences for multi-step prediction evaluation.
        
        Returns sequences of consecutive observations from test data.
        
        Args:
            num_sequences: Number of test sequences to return
            max_length: Maximum length of each sequence
            
        Returns:
            Tuple of (initial_states, future_states)
            - initial_states: [num_sequences, observation_size]
            - future_states: [max_length, num_sequences, observation_size]
        """
        test_data = self.test_dataset.data  # [n_samples, observation_size]
        n_samples = len(test_data)
        
        # Limit sequence length to available data
        actual_length = min(max_length, n_samples - 1)
        actual_num_seq = min(num_sequences, n_samples - actual_length)
        
        if actual_num_seq <= 0:
            raise ValueError(f"Not enough test data for {num_sequences} sequences of length {max_length}")
        
        # Sample starting indices evenly across the test set
        step = (n_samples - actual_length) // actual_num_seq
        start_indices = [i * step for i in range(actual_num_seq)]
        
        initial_states = []
        future_sequences = []
        
        for start_idx in start_indices:
            initial_states.append(test_data[start_idx])
            future_seq = test_data[start_idx + 1 : start_idx + 1 + actual_length]
            future_sequences.append(future_seq)
        
        initial_states = torch.stack(initial_states, dim=0)  # [num_seq, obs_size]
        future_states = torch.stack(future_sequences, dim=1)  # [length, num_seq, obs_size]
        
        return initial_states, future_states
    
    def extract_current_returns(self, observations: torch.Tensor) -> torch.Tensor:
        """Extract the current (most recent) log-returns from embedded observations.
        
        Given an embedded observation Y_t = [y_t, y_{t-1}, ..., y_{t-d+1}],
        extract just y_t (the first n_assets elements).
        
        Args:
            observations: Tensor of shape [..., embedding_dim * n_assets]
            
        Returns:
            Tensor of shape [..., n_assets] containing current log-returns
        """
        return observations[..., :self.n_assets]
    
    def destandardize_returns(self, standardized: torch.Tensor) -> torch.Tensor:
        """Convert standardized log-returns back to original scale.
        
        Args:
            standardized: Standardized log-returns [..., n_assets]
            
        Returns:
            Original-scale log-returns [..., n_assets]
        """
        mean = torch.from_numpy(self.stats.mean).float().to(standardized.device)
        std = torch.from_numpy(self.stats.std).float().to(standardized.device)
        return standardized * std + mean


def create_finance_env(
    config: Optional[FinanceDataConfig] = None,
    from_config: Optional["Config"] = None,
    sequence_length: Optional[int] = None,
) -> FinanceEnv:
    """Create a FinanceEnv from configuration.
    
    Args:
        config: FinanceDataConfig (if provided directly)
        from_config: Full Config object (extracts FINANCE and TRAIN settings)
        sequence_length: Override sequence length (default: from config or 1)
        
    Returns:
        FinanceEnv instance ready for training
    """
    if from_config is not None:
        # Extract finance settings from full Config
        finance_cfg = from_config.ENV.FINANCE
        config = FinanceDataConfig(
            tickers=finance_cfg.TICKERS,
            start_date=finance_cfg.START_DATE,
            end_date=finance_cfg.END_DATE,
            train_end=finance_cfg.TRAIN_END,
            val_end=finance_cfg.VAL_END,
            embedding_dim=finance_cfg.EMBEDDING_DIM,
            cache_dir=finance_cfg.CACHE_DIR,
        )
        # Use TRAIN.SEQUENCE_LENGTH if not overridden (for sequence training)
        if sequence_length is None:
            sequence_length = from_config.TRAIN.SEQUENCE_LENGTH if from_config.TRAIN.USE_SEQUENCE_LOSS else 1
    
    if config is None:
        config = FinanceDataConfig()
    
    # Default to pairwise (sequence_length=1) if not specified
    if sequence_length is None:
        sequence_length = 1
    
    # Load data with appropriate sequence length
    train_ds, val_ds, test_ds, stats, metadata = load_finance_data(config, sequence_length=sequence_length)
    
    return FinanceEnv(
        train_dataset=train_ds,
        val_dataset=val_ds,
        test_dataset=test_ds,
        stats=stats,
        metadata=metadata,
    )

