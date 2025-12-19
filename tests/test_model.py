"""Unit tests for PyTorch Koopman models."""

import torch
import pytest
from config import get_config, Config
from model import (
    MLPCoder,
    GenericKM,
    make_model,
    shrink,
    get_activation
)


class TestUtilityFunctions:
    """Test utility functions."""
    
    def test_shrink_positive(self):
        """Test soft thresholding on positive values."""
        x = torch.tensor([2.0, 1.5, 0.5])
        threshold = 1.0
        result = shrink(x, threshold)
        expected = torch.tensor([1.0, 0.5, 0.0])
        assert torch.allclose(result, expected)
    
    def test_shrink_negative(self):
        """Test soft thresholding on negative values."""
        x = torch.tensor([-2.0, -1.5, -0.5])
        threshold = 1.0
        result = shrink(x, threshold)
        expected = torch.tensor([-1.0, -0.5, 0.0])
        assert torch.allclose(result, expected)
    
    def test_shrink_mixed(self):
        """Test soft thresholding on mixed values."""
        x = torch.tensor([2.0, -1.5, 0.5, -0.3])
        threshold = 1.0
        result = shrink(x, threshold)
        expected = torch.tensor([1.0, -0.5, 0.0, 0.0])
        assert torch.allclose(result, expected)
    
    def test_get_activation_relu(self):
        """Test ReLU activation retrieval."""
        act = get_activation('relu')
        assert isinstance(act, torch.nn.ReLU)
    
    def test_get_activation_tanh(self):
        """Test Tanh activation retrieval."""
        act = get_activation('tanh')
        assert isinstance(act, torch.nn.Tanh)
    
    def test_get_activation_gelu(self):
        """Test GELU activation retrieval."""
        act = get_activation('gelu')
        assert isinstance(act, torch.nn.GELU)
    
    def test_get_activation_invalid(self):
        """Test invalid activation name."""
        with pytest.raises(ValueError):
            get_activation('invalid_activation')


class TestMLPCoder:
    """Test MLPCoder module."""
    
    def test_initialization(self):
        """Test MLPCoder can be initialized."""
        coder = MLPCoder(
            input_size=10,
            target_size=5,
            hidden_layers=[16, 16],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        assert coder.input_size == 10
        assert coder.target_size == 5
        assert len(coder.hidden_layers) == 2
    
    def test_forward_shape(self):
        """Test forward pass output shape."""
        coder = MLPCoder(
            input_size=10,
            target_size=5,
            hidden_layers=[16, 16],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        x = torch.randn(32, 10)
        y = coder(x)
        assert y.shape == (32, 5)
    
    def test_forward_batch_independence(self):
        """Test that batch elements are processed independently."""
        coder = MLPCoder(
            input_size=5,
            target_size=3,
            hidden_layers=[8],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        x1 = torch.randn(1, 5)
        x2 = torch.randn(1, 5)
        x_batch = torch.cat([x1, x2], dim=0)
        
        y1 = coder(x1)
        y2 = coder(x2)
        y_batch = coder(x_batch)
        
        assert torch.allclose(y_batch[0], y1[0], atol=1e-6)
        assert torch.allclose(y_batch[1], y2[0], atol=1e-6)
    
    def test_last_relu(self):
        """Test last_relu option applies ReLU to output."""
        coder_with_relu = MLPCoder(
            input_size=5,
            target_size=3,
            hidden_layers=[],
            last_relu=True,
            use_bias=False,
            activation='relu'
        )
        coder_without_relu = MLPCoder(
            input_size=5,
            target_size=3,
            hidden_layers=[],
            last_relu=False,
            use_bias=False,
            activation='relu'
        )
        
        # Use same weights
        coder_with_relu.network[0].weight.data = coder_without_relu.network[0].weight.data.clone()
        
        x = torch.randn(1, 5)
        y_with = coder_with_relu(x)
        y_without = coder_without_relu(x)
        
        # Output with ReLU should be non-negative
        assert torch.all(y_with >= 0)
        # Outputs should match after applying ReLU
        assert torch.allclose(y_with, torch.relu(y_without))


class TestFinanceSparseKM:
    """Test finance sparse Koopman model."""
    
    def test_initialization(self):
        cfg = get_config("finance_sparse")
        obs_size = 120
        model = GenericKM(cfg, obs_size)
        assert model.observation_size == obs_size
        assert model.target_size == cfg.MODEL.TARGET_SIZE
    
    def test_encode_decode_shape(self):
        """Test encode and decode output shapes."""
        cfg = get_config("finance_sparse")
        cfg.MODEL.TARGET_SIZE = 1024
        obs_size = 120
        batch_size = 16
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(batch_size, obs_size)
        
        # Test encode
        z = model.encode(x)
        assert z.shape == (batch_size, cfg.MODEL.TARGET_SIZE)
        
        # Test decode
        x_recon = model.decode(z)
        assert x_recon.shape == (batch_size, obs_size)
    
    def test_reconstruction(self):
        """Test reconstruction method."""
        cfg = get_config("finance_sparse")
        cfg.MODEL.TARGET_SIZE = 1024
        obs_size = 120
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        x_recon = model.reconstruction(x)
        assert x_recon.shape == x.shape
    
    def test_kmatrix_shape(self):
        """Test Koopman matrix has correct shape."""
        cfg = get_config("finance_sparse")
        cfg.MODEL.TARGET_SIZE = 1024
        obs_size = 120
        
        model = GenericKM(cfg, obs_size)
        kmat = model.kmatrix()
        assert kmat.shape == (1024, 1024)
    
    def test_step_latent(self):
        """Test stepping in latent space."""
        cfg = get_config("finance_sparse")
        cfg.MODEL.TARGET_SIZE = 1024
        obs_size = 120
        
        model = GenericKM(cfg, obs_size)
        y = torch.randn(4, 1024)
        ny = model.step_latent(y)
        assert ny.shape == y.shape
    
    def test_step_env(self):
        """Test stepping in observation space via Koopman operator."""
        cfg = get_config("finance_sparse")
        cfg.MODEL.TARGET_SIZE = 1024
        obs_size = 120
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        nx_pred = model.step_env(x)
        assert nx_pred.shape == x.shape
    
    def test_residual(self):
        """Test residual computation."""
        cfg = get_config("finance_sparse")
        cfg.MODEL.TARGET_SIZE = 1024
        obs_size = 120
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        residual = model.residual(x, nx)
        assert residual.shape == (8,)
        assert torch.all(residual >= 0)  # Norm is non-negative
    
    def test_sparsity_loss(self):
        """Test sparsity loss computation."""
        cfg = get_config("finance_sparse")
        obs_size = 120
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        loss = model.sparsity_loss(x)
        assert loss.ndim == 0  # Scalar
        assert loss >= 0  # L1 norm is non-negative
    
    def test_loss_computation(self):
        """Test full loss computation."""
        cfg = get_config("finance_sparse")
        cfg.MODEL.TARGET_SIZE = 1024
        obs_size = 120
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(8, obs_size)
        nx = torch.randn(8, obs_size)
        
        loss, metrics = model.loss(x, nx)
        
        # Check loss is scalar
        assert loss.ndim == 0
        
        # Check metrics
        assert 'loss' in metrics
        assert 'residual_loss' in metrics
        assert 'reconst_loss' in metrics
        assert 'prediction_loss' in metrics
        assert 'sparsity_loss' in metrics
        assert 'A_max_eigenvalue' in metrics
        assert 'sparsity_ratio' in metrics
    
    def test_norm_fn_id(self):
        """Test identity normalization function."""
        cfg = get_config("finance_sparse")
        cfg.MODEL.NORM_FN = "id"
        obs_size = 120
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(4, obs_size)
        z = model.encode(x)
        
        # With identity norm, we just check it doesn't crash
        assert z.shape == (4, cfg.MODEL.TARGET_SIZE)


class TestModelFactory:
    """Test model factory function."""
    
    def test_make_model_finance_sparse(self):
        """Test creating SparseKM (alias for GenericKM) via factory."""
        cfg = get_config("finance_sparse")
        obs_size = 120
        model = make_model(cfg, obs_size)
        assert isinstance(model, GenericKM)
    
    def test_make_model_invalid(self):
        """Test factory raises error for invalid model name."""
        cfg = get_config("finance_sparse")
        cfg.MODEL.MODEL_NAME = "InvalidModel"
        obs_size = 120
        
        with pytest.raises(ValueError):
            make_model(cfg, obs_size)


class TestGradientFlow:
    """Test gradient flow through models."""
    
    def test_generic_km_gradients(self):
        """Test gradients flow through GenericKM."""
        cfg = get_config("finance_sparse")
        cfg.MODEL.TARGET_SIZE = 1024
        obs_size = 120
        
        model = GenericKM(cfg, obs_size)
        x = torch.randn(64, obs_size, requires_grad=True)
        nx = torch.randn(64, obs_size)
        
        loss, _ = model.loss(x, nx)
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert model.encoder.network[0].weight.grad is not None
        assert model.kmat.grad is not None
    
if __name__ == "__main__":
    pytest.main([__file__, "-v"])

