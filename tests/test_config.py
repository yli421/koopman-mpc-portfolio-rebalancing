"""Tests for the config system (config.py).

Tests configuration loading, modification, and registry functionality.
"""

import pytest
import os
import json

from config import (
    get_config,
    get_default_config,
    get_train_finance_sparse_config,
    Config
)


def test_get_default_config():
    """Test that default config has expected structure and values."""
    cfg = get_default_config()
    
    # Check top-level keys
    assert hasattr(cfg, "SEED")
    assert hasattr(cfg, "ENV")
    assert hasattr(cfg, "MODEL")
    assert hasattr(cfg, "TRAIN")
    
    # Check ENV structure
    assert hasattr(cfg.ENV, "ENV_NAME")
    assert hasattr(cfg.ENV, "FINANCE")
    
    # Check MODEL structure
    assert hasattr(cfg.MODEL, "MODEL_NAME")
    assert hasattr(cfg.MODEL, "TARGET_SIZE")
    assert hasattr(cfg.MODEL, "ENCODER")
    assert hasattr(cfg.MODEL, "DECODER")
    
    # Check loss coefficients
    assert hasattr(cfg.MODEL, "RES_COEFF")
    assert hasattr(cfg.MODEL, "RECONST_COEFF")
    assert hasattr(cfg.MODEL, "PRED_COEFF")
    assert hasattr(cfg.MODEL, "SPARSITY_COEFF")
    
    # Check TRAIN structure
    assert hasattr(cfg.TRAIN, "NUM_STEPS")
    assert hasattr(cfg.TRAIN, "BATCH_SIZE")
    assert hasattr(cfg.TRAIN, "LR")


def test_get_named_configs():
    """Test that named configurations load correctly."""
    # Test finance_sparse config
    cfg_finance = get_train_finance_sparse_config()
    assert cfg_finance.ENV.ENV_NAME == "finance"
    assert cfg_finance.MODEL.MODEL_NAME == "GenericKM"
    assert cfg_finance.MODEL.TARGET_SIZE == 1024


def test_config_registry():
    """Test that config registry works."""
    cfg = get_config("default")
    assert cfg is not None
    
    cfg_finance = get_config("finance_sparse")
    assert cfg_finance.ENV.ENV_NAME == "finance"
    
    with pytest.raises(ValueError):
        get_config("nonexistent")


def test_config_modification():
    """Test that config can be modified."""
    cfg = get_default_config()
    original_lr = cfg.TRAIN.LR
    
    cfg.TRAIN.LR = 42
    assert cfg.TRAIN.LR == 42
    assert cfg.TRAIN.LR != original_lr


def test_config_serialization(tmp_path):
    """Test JSON serialization and deserialization."""
    cfg = get_default_config()
    cfg.SEED = 42
    cfg.TRAIN.LR = 0.005
    
    # Save to JSON
    json_path = tmp_path / "config.json"
    cfg.to_json(str(json_path))
    
    # Load from JSON
    new_cfg = Config.from_json(str(json_path))
    
    assert new_cfg.SEED == 42
    assert new_cfg.TRAIN.LR == 0.005
    assert new_cfg.ENV.ENV_NAME == cfg.ENV.ENV_NAME
