"""Unit tests for data_finance.py finance data module.

Tests cover:
- Log-return computation correctness
- Time-delay embedding shape and shift property
- Chronological split boundaries (leak-free)
- Standardization using training-only stats
- Dataset iteration and shapes
"""

import unittest
import numpy as np
import pandas as pd
import torch
from datetime import datetime

from data_finance import (
    FinanceDataConfig,
    FinanceStats,
    compute_log_returns,
    compute_standardization_stats,
    standardize_returns,
    time_delay_embedding,
    create_finance_splits,
    verify_embedding_shift,
    FinanceDataset,
    clean_price_data,
    compute_return_stats,
    compute_autocorrelation,
)


class TestLogReturns(unittest.TestCase):
    """Test log-return computation."""

    def test_log_returns_basic(self):
        """Test basic log-return computation."""
        # Create simple price series
        dates = pd.date_range("2020-01-01", periods=5, freq="D")
        prices = pd.DataFrame({
            "AAPL": [100.0, 110.0, 105.0, 115.0, 120.0],
            "MSFT": [200.0, 210.0, 220.0, 215.0, 225.0],
        }, index=dates)
        
        log_returns = compute_log_returns(prices)
        
        # Should have one less row (first row is NaN and dropped)
        self.assertEqual(len(log_returns), 4)
        self.assertEqual(list(log_returns.columns), ["AAPL", "MSFT"])
        
        # Verify manual calculation for first return
        expected_aapl = np.log(110.0 / 100.0)
        expected_msft = np.log(210.0 / 200.0)
        self.assertAlmostEqual(log_returns["AAPL"].iloc[0], expected_aapl, places=6)
        self.assertAlmostEqual(log_returns["MSFT"].iloc[0], expected_msft, places=6)
    
    def test_log_returns_shape(self):
        """Test that log-returns have correct shape."""
        n_days = 100
        n_assets = 5
        dates = pd.date_range("2020-01-01", periods=n_days, freq="D")
        prices = pd.DataFrame(
            np.random.uniform(50, 200, (n_days, n_assets)),
            index=dates,
            columns=[f"STOCK_{i}" for i in range(n_assets)]
        )
        
        log_returns = compute_log_returns(prices)
        
        self.assertEqual(log_returns.shape, (n_days - 1, n_assets))


class TestStandardization(unittest.TestCase):
    """Test standardization using training data only."""

    def setUp(self):
        """Create test data with known statistics."""
        dates = pd.date_range("2018-01-01", periods=200, freq="D")
        np.random.seed(42)
        self.log_returns = pd.DataFrame({
            "AAPL": np.random.normal(0.001, 0.02, 200),
            "MSFT": np.random.normal(0.0005, 0.015, 200),
        }, index=dates)
        self.train_end = "2018-06-01"
    
    def test_stats_from_training_only(self):
        """Test that statistics are computed from training data only."""
        stats = compute_standardization_stats(self.log_returns, self.train_end)
        
        train_data = self.log_returns[self.log_returns.index <= self.train_end]
        
        np.testing.assert_array_almost_equal(
            stats.mean, train_data.mean().values, decimal=6
        )
        np.testing.assert_array_almost_equal(
            stats.std, train_data.std().values, decimal=6
        )
    
    def test_standardization_zero_mean_unit_var_on_train(self):
        """Test that standardized training data has ~zero mean and ~unit variance."""
        stats = compute_standardization_stats(self.log_returns, self.train_end)
        standardized = standardize_returns(self.log_returns, stats)
        
        train_standardized = standardized[standardized.index <= self.train_end]
        
        # Mean should be approximately 0
        np.testing.assert_array_almost_equal(
            train_standardized.mean().values, 
            np.zeros(2), 
            decimal=5
        )
        
        # Std should be approximately 1
        np.testing.assert_array_almost_equal(
            train_standardized.std().values, 
            np.ones(2), 
            decimal=1  # Less strict due to sample variance
        )
    
    def test_stats_has_tickers(self):
        """Test that stats object contains ticker names."""
        stats = compute_standardization_stats(self.log_returns, self.train_end)
        self.assertEqual(stats.tickers, ["AAPL", "MSFT"])


class TestTimeDelayEmbedding(unittest.TestCase):
    """Test time-delay embedding construction."""

    def test_embedding_shape(self):
        """Test embedding output shape."""
        T, n_assets = 100, 5
        embedding_dim = 5
        data = np.random.randn(T, n_assets).astype(np.float32)
        
        embedded = time_delay_embedding(data, embedding_dim)
        
        expected_samples = T - embedding_dim + 1
        expected_features = embedding_dim * n_assets
        self.assertEqual(embedded.shape, (expected_samples, expected_features))
    
    def test_embedding_shift_property(self):
        """Test that Y_{t+1} is a shifted version of Y_t plus new observation."""
        T, n_assets = 50, 3
        embedding_dim = 4
        data = np.random.randn(T, n_assets).astype(np.float32)
        
        embedded = time_delay_embedding(data, embedding_dim)
        
        # Use utility function to verify shift property
        is_valid = verify_embedding_shift(embedded, n_assets, embedding_dim)
        self.assertTrue(is_valid, "Time-delay embedding shift property violated")
    
    def test_embedding_content_correctness(self):
        """Test that embedding contains correct values."""
        # Create simple sequential data for easy verification
        T, n_assets = 10, 2
        embedding_dim = 3
        data = np.arange(T * n_assets).reshape(T, n_assets).astype(np.float32)
        
        embedded = time_delay_embedding(data, embedding_dim)
        
        # First embedded observation Y_0 should be [y_2, y_1, y_0] (most recent first)
        # where y_i is row i of data
        # Y_0 = [data[2], data[1], data[0]] flattened
        expected_first = np.concatenate([data[2], data[1], data[0]])
        np.testing.assert_array_almost_equal(embedded[0], expected_first)
        
        # Second embedded observation Y_1 should be [y_3, y_2, y_1]
        expected_second = np.concatenate([data[3], data[2], data[1]])
        np.testing.assert_array_almost_equal(embedded[1], expected_second)
    
    def test_embedding_too_short_data(self):
        """Test that embedding raises error for too short data."""
        data = np.random.randn(3, 2).astype(np.float32)
        embedding_dim = 5
        
        with self.assertRaises(ValueError):
            time_delay_embedding(data, embedding_dim)


class TestChronologicalSplits(unittest.TestCase):
    """Test chronological train/val/test splitting."""

    def setUp(self):
        """Create test data spanning multiple years."""
        # Create 4 years of daily data
        dates = pd.date_range("2016-01-01", "2019-12-31", freq="B")  # Business days
        np.random.seed(42)
        n_assets = 3
        self.log_returns = pd.DataFrame(
            np.random.randn(len(dates), n_assets) * 0.02,
            index=dates,
            columns=["A", "B", "C"]
        )
        self.train_end = "2017-12-31"
        self.val_end = "2018-12-31"
        self.embedding_dim = 5
    
    def test_split_boundaries_correct(self):
        """Test that split boundaries are respected."""
        stats = compute_standardization_stats(self.log_returns, self.train_end)
        
        train_data, train_dates, val_data, val_dates, test_data, test_dates = \
            create_finance_splits(
                self.log_returns, stats, self.train_end, self.val_end, self.embedding_dim
            )
        
        # All train dates should be <= train_end
        self.assertTrue(all(d <= pd.Timestamp(self.train_end) for d in train_dates))
        
        # All val dates should be in (train_end, val_end]
        self.assertTrue(all(
            pd.Timestamp(self.train_end) < d <= pd.Timestamp(self.val_end) 
            for d in val_dates
        ))
        
        # All test dates should be > val_end
        self.assertTrue(all(d > pd.Timestamp(self.val_end) for d in test_dates))
    
    def test_no_data_leakage(self):
        """Test that train/val/test have no overlapping dates."""
        stats = compute_standardization_stats(self.log_returns, self.train_end)
        
        train_data, train_dates, val_data, val_dates, test_data, test_dates = \
            create_finance_splits(
                self.log_returns, stats, self.train_end, self.val_end, self.embedding_dim
            )
        
        train_set = set(train_dates)
        val_set = set(val_dates)
        test_set = set(test_dates)
        
        # No overlaps
        self.assertEqual(len(train_set & val_set), 0)
        self.assertEqual(len(train_set & test_set), 0)
        self.assertEqual(len(val_set & test_set), 0)
    
    def test_split_data_shapes(self):
        """Test that split data has correct shapes."""
        stats = compute_standardization_stats(self.log_returns, self.train_end)
        n_assets = len(self.log_returns.columns)
        
        train_data, train_dates, val_data, val_dates, test_data, test_dates = \
            create_finance_splits(
                self.log_returns, stats, self.train_end, self.val_end, self.embedding_dim
            )
        
        expected_features = self.embedding_dim * n_assets
        
        self.assertEqual(train_data.shape[1], expected_features)
        self.assertEqual(val_data.shape[1], expected_features)
        self.assertEqual(test_data.shape[1], expected_features)
        
        # Number of samples should match dates
        self.assertEqual(len(train_data), len(train_dates))
        self.assertEqual(len(val_data), len(val_dates))
        self.assertEqual(len(test_data), len(test_dates))
    
    def test_standardization_uses_train_only(self):
        """Test that standardization is applied using training stats only."""
        # Create data where train and test have different means
        dates = pd.date_range("2016-01-01", "2019-12-31", freq="B")
        data = pd.DataFrame(index=dates)
        
        # Train period has mean ~0.01, test period has mean ~0.05
        train_mask = dates <= self.train_end
        data["A"] = 0.0
        data.loc[train_mask, "A"] = np.random.normal(0.01, 0.02, train_mask.sum())
        data.loc[~train_mask, "A"] = np.random.normal(0.05, 0.02, (~train_mask).sum())
        
        stats = compute_standardization_stats(data, self.train_end)
        
        # Stats should reflect training data only
        train_data = data[data.index <= self.train_end]
        self.assertAlmostEqual(stats.mean[0], train_data["A"].mean(), places=5)


class TestFinanceDataset(unittest.TestCase):
    """Test PyTorch Dataset for finance data."""

    def setUp(self):
        """Create test embedded data."""
        np.random.seed(42)
        self.n_samples = 100
        self.embedding_size = 20  # 4 assets * 5 embedding_dim
        self.data = np.random.randn(self.n_samples, self.embedding_size).astype(np.float32)
    
    def test_pairwise_dataset_length(self):
        """Test pairwise dataset has correct length."""
        dataset = FinanceDataset(self.data, sequence_length=1)
        
        # Pairwise: we lose 1 sample (need pairs)
        self.assertEqual(len(dataset), self.n_samples - 1)
    
    def test_pairwise_dataset_shapes(self):
        """Test pairwise dataset returns correct shapes."""
        dataset = FinanceDataset(self.data, sequence_length=1)
        
        y_t, y_t1 = dataset[0]
        
        self.assertEqual(y_t.shape, (self.embedding_size,))
        self.assertEqual(y_t1.shape, (self.embedding_size,))
        self.assertTrue(torch.is_tensor(y_t))
        self.assertTrue(torch.is_tensor(y_t1))
    
    def test_pairwise_dataset_consecutive(self):
        """Test pairwise samples are consecutive."""
        dataset = FinanceDataset(self.data, sequence_length=1)
        
        for i in range(5):
            y_t, y_t1 = dataset[i]
            
            # y_t should be data[i], y_t1 should be data[i+1]
            np.testing.assert_array_almost_equal(
                y_t.numpy(), self.data[i]
            )
            np.testing.assert_array_almost_equal(
                y_t1.numpy(), self.data[i + 1]
            )
    
    def test_sequence_dataset_length(self):
        """Test sequence dataset has correct length."""
        seq_len = 10
        dataset = FinanceDataset(self.data, sequence_length=seq_len)
        
        # We lose seq_len samples
        self.assertEqual(len(dataset), self.n_samples - seq_len)
    
    def test_sequence_dataset_shapes(self):
        """Test sequence dataset returns correct shapes."""
        seq_len = 10
        dataset = FinanceDataset(self.data, sequence_length=seq_len)
        
        sequence = dataset[0]
        
        # Should be [seq_len + 1, embedding_size]
        self.assertEqual(sequence.shape, (seq_len + 1, self.embedding_size))
        self.assertTrue(torch.is_tensor(sequence))
    
    def test_sequence_dataset_consecutive(self):
        """Test sequence samples are consecutive."""
        seq_len = 5
        dataset = FinanceDataset(self.data, sequence_length=seq_len)
        
        sequence = dataset[3]
        
        # sequence[j] should be data[3 + j]
        for j in range(seq_len + 1):
            np.testing.assert_array_almost_equal(
                sequence[j].numpy(), self.data[3 + j]
            )
    
    def test_observation_size_property(self):
        """Test observation_size property."""
        dataset = FinanceDataset(self.data, sequence_length=1)
        self.assertEqual(dataset.observation_size, self.embedding_size)
    
    def test_dataset_too_short(self):
        """Test that dataset raises error for too short data."""
        short_data = np.random.randn(5, 10).astype(np.float32)
        
        with self.assertRaises(ValueError):
            FinanceDataset(short_data, sequence_length=10)


class TestDataCleaning(unittest.TestCase):
    """Test data cleaning functions."""

    def test_clean_price_data_drops_high_missing(self):
        """Test that assets with too much missing data are dropped."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        prices = pd.DataFrame({
            "GOOD": np.random.uniform(100, 200, 100),
            "BAD": np.concatenate([np.random.uniform(100, 200, 50), [np.nan] * 50]),
        }, index=dates)
        
        cleaned = clean_price_data(prices, max_missing_ratio=0.3)
        
        self.assertIn("GOOD", cleaned.columns)
        self.assertNotIn("BAD", cleaned.columns)
    
    def test_clean_price_data_forward_fills(self):
        """Test that short gaps are forward-filled."""
        dates = pd.date_range("2020-01-01", periods=10, freq="D")
        prices = pd.DataFrame({
            "A": [100.0, np.nan, np.nan, 103.0, 104.0, 105.0, 106.0, 107.0, 108.0, 109.0],
        }, index=dates)
        
        # Use max_missing_ratio=0.3 to keep asset A (20% missing < 30% threshold)
        cleaned = clean_price_data(prices, max_missing_ratio=0.3, max_gap_days=5)
        
        # Should have no NaN after forward fill
        self.assertFalse(cleaned.isna().any().any())
        # Filled values should be the last valid value
        self.assertEqual(cleaned["A"].iloc[1], 100.0)
        self.assertEqual(cleaned["A"].iloc[2], 100.0)


class TestUtilityFunctions(unittest.TestCase):
    """Test utility functions for analysis."""

    def test_compute_return_stats(self):
        """Test return statistics computation."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        log_returns = pd.DataFrame({
            "A": np.random.normal(0.001, 0.02, 100),
            "B": np.random.normal(-0.001, 0.03, 100),
        }, index=dates)
        
        stats = compute_return_stats(log_returns)
        
        self.assertIn("mean", stats.columns)
        self.assertIn("std", stats.columns)
        self.assertIn("skew", stats.columns)
        self.assertIn("kurtosis", stats.columns)
        self.assertEqual(len(stats), 2)
    
    def test_compute_autocorrelation(self):
        """Test autocorrelation computation."""
        dates = pd.date_range("2020-01-01", periods=100, freq="D")
        log_returns = pd.DataFrame({
            "A": np.random.randn(100),
            "B": np.random.randn(100),
        }, index=dates)
        
        autocorr = compute_autocorrelation(log_returns, lag=1)
        
        self.assertEqual(len(autocorr), 2)
        # Autocorrelation should be in [-1, 1]
        self.assertTrue(all(-1 <= x <= 1 for x in autocorr))


class TestEmbeddingShiftVerification(unittest.TestCase):
    """Test the embedding shift verification utility."""

    def test_verify_correct_embedding(self):
        """Test that verification passes for correct embedding."""
        T, n_assets = 30, 3
        embedding_dim = 4
        data = np.random.randn(T, n_assets).astype(np.float32)
        
        embedded = time_delay_embedding(data, embedding_dim)
        
        is_valid = verify_embedding_shift(embedded, n_assets, embedding_dim)
        self.assertTrue(is_valid)
    
    def test_verify_incorrect_embedding(self):
        """Test that verification fails for incorrect embedding."""
        n_assets = 3
        embedding_dim = 4
        # Create random data that doesn't satisfy the shift property
        embedded = np.random.randn(10, embedding_dim * n_assets).astype(np.float32)
        
        is_valid = verify_embedding_shift(embedded, n_assets, embedding_dim)
        self.assertFalse(is_valid)


if __name__ == "__main__":
    unittest.main()

