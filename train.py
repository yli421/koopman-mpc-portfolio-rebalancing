"""
Training script for Koopman Autoencoder models - Finance Version.

This script provides a complete training pipeline for learning Koopman operator
representations of financial market dynamics using PyTorch.

Usage:
    python train.py --num_steps 5000
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm

print("Loading torch...")
import torch
import torch.nn as nn
print("Torch loaded.")

print("Loading config...")
from config import Config, get_config
print("Config loaded.")

print("Loading data...")
from data_finance import create_finance_env, FinanceEnv
print("Data loaded.")

print("Loading model...")
from model import make_model
print("Model loaded.")

# Lazy import evaluation - only load when needed
print("All core imports loaded.")


class MetricsLogger:
    """Simple file-based metrics logger.
    
    Logs metrics to JSON files for later analysis or plotting.
    Uses buffered writes to reduce I/O overhead.
    """
    
    def __init__(self, log_dir: Path, flush_interval: int = 100):
        self.log_dir = log_dir
        self.metrics_file = log_dir / 'metrics_history.jsonl'
        self.metrics_history: List[Dict] = []
        self.buffer: List[str] = []
        self.flush_interval = flush_interval
        self.step_count = 0
    
    def log_scalar(self, name: str, value: float, step: int):
        """Log a scalar metric."""
        entry = {
            'step': step,
            'name': name,
            'value': value,
        }
        # Buffer writes to reduce I/O overhead
        self.buffer.append(json.dumps(entry) + '\n')
        self.metrics_history.append(entry)
        self.step_count += 1
        
        # Flush buffer periodically
        if len(self.buffer) >= self.flush_interval:
            self.flush()
    
    def flush(self):
        """Flush buffered metrics to disk."""
        if self.buffer:
            with open(self.metrics_file, 'a') as f:
                f.writelines(self.buffer)
            self.buffer.clear()
    
    def log_dict(self, metrics: Dict[str, float], step: int, prefix: str = ''):
        """Log a dictionary of metrics."""
        for key, value in metrics.items():
            name = f"{prefix}/{key}" if prefix else key
            self.log_scalar(name, value, step)
    
    def close(self):
        """Save final summary and flush any remaining buffered writes."""
        # Flush any remaining buffered metrics
        self.flush()
        
        summary_file = self.log_dir / 'metrics_summary.json'
        
        # Compute summary statistics
        summary = {}
        metrics_by_name = {}
        for entry in self.metrics_history:
            name = entry['name']
            if name not in metrics_by_name:
                metrics_by_name[name] = []
            metrics_by_name[name].append(entry['value'])
        
        for name, values in metrics_by_name.items():
            summary[name] = {
                'final': values[-1] if values else None,
                'min': min(values) if values else None,
                'max': max(values) if values else None,
                'mean': sum(values) / len(values) if values else None,
            }
        
        with open(summary_file, 'w') as f:
            json.dump(summary, f, indent=2)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    nx: torch.Tensor,
    cfg: Config,
    dt: float,
) -> Dict[str, float]:
    """Perform one training step.
    
    Args:
        model: Koopman machine model
        optimizer: PyTorch optimizer
        x: Current states [batch_size, observation_size] OR
           sequence [batch_size, seq_len, observation_size] if USE_SEQUENCE_LOSS=True
        nx: Next states [batch_size, observation_size] (unused if USE_SEQUENCE_LOSS=True)
        cfg: Configuration object
        dt: Time step for ODE integration
        
    Returns:
        Dictionary of metrics
    """
    model.train()
    optimizer.zero_grad()
    
    # Compute loss
    if cfg.TRAIN.USE_SEQUENCE_LOSS:
        # x is a sequence: [batch_size, seq_len, observation_size]
        loss, metrics = model.loss_sequence(x, dt)
    else:
        # Standard single-step loss
        loss, metrics = model.loss(x, nx)
    
    # Backward pass
    loss.backward()
    optimizer.step()

    return metrics


def build_optimizer(model: nn.Module, cfg: Config) -> torch.optim.Optimizer:
    """Create optimizer with a specific learning rate for the Koopman matrix.
    
    This constructs parameter groups so that parameters named with 'kmat' use
    cfg.TRAIN.K_MATRIX_LR while all other parameters use cfg.TRAIN.LR.
    """
    kmat_params = []
    other_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'kmat' in name:
            kmat_params.append(param)
        else:
            other_params.append(param)

    param_groups = []
    if other_params:
        param_groups.append({
            'params': other_params,
            'lr': cfg.TRAIN.LR,
            'weight_decay': cfg.TRAIN.WEIGHT_DECAY,
        })
    if kmat_params:
        param_groups.append({
            'params': kmat_params,
            'lr': cfg.TRAIN.K_MATRIX_LR,
            'weight_decay': 0.0,  # No weight decay on Koopman matrix
        })

    return torch.optim.AdamW(param_groups)


def evaluate_finance(
    model: nn.Module,
    initial_states: torch.Tensor,
    future_states: torch.Tensor,
    max_horizon: int = 50,
    periodic_reencode_periods: List[int] = [5, 10, 25],
) -> Dict[str, Any]:
    """Evaluate model on finance test data with multi-step prediction.
    
    Args:
        model: Trained Koopman model
        initial_states: Initial observations [batch, obs_size]
        future_states: Ground truth future [horizon, batch, obs_size]
        max_horizon: Maximum prediction horizon to evaluate
        periodic_reencode_periods: List of periods for periodic re-encoding
        
    Returns:
        Dictionary with evaluation metrics for all rollout modes
    """
    from evaluation import rollout_every_step_reencode, rollout_no_reencode, rollout_periodic_reencode
    
    model.eval()
    device = next(model.parameters()).device
    
    horizon = min(max_horizon, future_states.shape[0])
    
    with torch.no_grad():
        initial_states = initial_states.to(device)
        true = future_states[:horizon].to(device)
        
        # Collect predictions from all modes
        predictions = {}
        mse_curves = {}
        l2_curves = {}
        
        # Predict with re-encoding at every step
        pred_reencode = rollout_every_step_reencode(model, initial_states, horizon)
        predictions['every_step'] = pred_reencode
        mse_curves['every_step'] = ((pred_reencode - true) ** 2).mean(dim=(1, 2))
        l2_curves['every_step'] = torch.norm(pred_reencode - true, dim=-1).mean(dim=1)
        
        # Predict without re-encoding (pure Koopman dynamics)
        pred_no_reencode = rollout_no_reencode(model, initial_states, horizon)
        predictions['no_reencode'] = pred_no_reencode
        mse_curves['no_reencode'] = ((pred_no_reencode - true) ** 2).mean(dim=(1, 2))
        l2_curves['no_reencode'] = torch.norm(pred_no_reencode - true, dim=-1).mean(dim=1)
        
        # Periodic re-encoding at different intervals
        for period in periodic_reencode_periods:
            mode_name = f'periodic_{period}'
            pred_periodic = rollout_periodic_reencode(model, initial_states, horizon, period=period)
            predictions[mode_name] = pred_periodic
            mse_curves[mode_name] = ((pred_periodic - true) ** 2).mean(dim=(1, 2))
            l2_curves[mode_name] = torch.norm(pred_periodic - true, dim=-1).mean(dim=1)
        
        # Find best periodic mode
        mean_mses = {mode: curve.mean().item() for mode, curve in mse_curves.items()}
        best_mode = min(mean_mses, key=mean_mses.get)
        
        return {
            # Individual mode curves for backward compatibility
            "mse_reencode": mse_curves['every_step'].cpu(),
            "mse_no_reencode": mse_curves['no_reencode'].cpu(),
            "l2_reencode": l2_curves['every_step'].cpu(),
            "l2_no_reencode": l2_curves['no_reencode'].cpu(),
            "mean_mse_reencode": mean_mses['every_step'],
            "mean_mse_no_reencode": mean_mses['no_reencode'],
            "final_mse_reencode": mse_curves['every_step'][-1].item(),
            "final_mse_no_reencode": mse_curves['no_reencode'][-1].item(),
            "pred_reencode": pred_reencode.cpu(),
            "pred_no_reencode": pred_no_reencode.cpu(),
            "true": true.cpu(),
            # All modes
            "mse_curves": {k: v.cpu() for k, v in mse_curves.items()},
            "l2_curves": {k: v.cpu() for k, v in l2_curves.items()},
            "mean_mses": mean_mses,
            "predictions": {k: v.cpu() for k, v in predictions.items()},
            "best_mode": best_mode,
            "best_mse": mean_mses[best_mode],
        }


def train(
    cfg: Config,
    log_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
) -> nn.Module:
    """Training function for finance data.
    
    Uses PyTorch DataLoader since finance data is pre-recorded.
    
    Args:
        cfg: Configuration object with ENV.ENV_NAME = "finance"
        log_dir: Directory for logs and checkpoints
        checkpoint_path: Path to checkpoint to resume from
        device: Device to train on
        
    Returns:
        Trained model
    """
    print("Initializing finance training...")
    
    # Setup logging directory and save config
    if log_dir is None:
        log_dir = './runs/kae_finance'
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = Path(log_dir) / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    cfg.to_json(str(run_dir / 'config.json'))
    
    logger = MetricsLogger(run_dir)
    
    print("Setting random seed...")
    torch.manual_seed(cfg.SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(cfg.SEED)
    
    print("Loading finance data...")
    finance_env = create_finance_env(from_config=cfg)
    
    # Create dataloaders
    train_loader = finance_env.get_dataloader(
        split='train',
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=True,
    )
    val_loader = finance_env.get_dataloader(
        split='val',
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
    )
    
    # Get test sequences for evaluation
    test_init, test_future = finance_env.get_test_sequences(
        num_sequences=min(100, len(finance_env.test_dataset) // 2),
        max_length=100,
    )
    
    print(f"Train samples: {len(finance_env.train_dataset)}")
    print(f"Val samples: {len(finance_env.val_dataset)}")
    print(f"Test samples: {len(finance_env.test_dataset)}")
    
    # For finance, dt is set to 1.0 (daily) since we don't use ODE integration
    dt = 1.0
    
    print("Creating model...")
    model = make_model(cfg, finance_env.observation_size)
    model = model.to(device)
    model.dt = dt
    
    print("Building optimizer...")
    optimizer = build_optimizer(model, cfg)
    
    start_step = 0
    start_epoch = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        start_epoch = checkpoint.get('epoch', 0)
        print(f"Resumed from checkpoint at step {start_step}")
    
    print(f"\nTraining {cfg.MODEL.MODEL_NAME} on finance data")
    print(f"Device: {device}")
    print(f"Observation size: {finance_env.observation_size}")
    print(f"  ({finance_env.n_assets} assets × {finance_env.embedding_dim} embedding_dim)")
    print(f"Target size (latent): {cfg.MODEL.TARGET_SIZE}")
    print(f"Batch size: {cfg.TRAIN.BATCH_SIZE}")
    print(f"Total steps: {cfg.TRAIN.NUM_STEPS}")
    print(f"Log directory: {run_dir}")
    print("-" * 80)
    
    best_eval_error = float('inf')
    global_step = start_step
    epoch = start_epoch
    
    # Initialize tqdm
    pbar = tqdm(total=cfg.TRAIN.NUM_STEPS, initial=start_step, desc="Training")
    
    # Training loop - iterate over epochs until we reach NUM_STEPS
    while global_step < cfg.TRAIN.NUM_STEPS:
        epoch += 1
        epoch_loss = 0.0
        epoch_batches = 0
        
        for batch in train_loader:
            if global_step >= cfg.TRAIN.NUM_STEPS:
                break
            
            # Unpack batch: for pairwise data, batch is (x, nx)
            if isinstance(batch, (list, tuple)):
                x, nx = batch
                x = x.to(device)
                nx = nx.to(device)
            else:
                # For sequence data, batch is a tensor
                x = batch.to(device)
                nx = None
            
            # Training step
            metrics = train_step(model, optimizer, x, nx, cfg, dt)
            logger.log_dict(metrics, global_step, prefix='train')
            
            epoch_loss += metrics['loss']
            epoch_batches += 1
            
            # Update progress bar
            pbar.update(1)
            pbar.set_postfix({
                'loss': f"{metrics['loss']:.4f}",
                'epoch': epoch,
            })

            # Print detailed metrics every 100 steps
            if global_step % 100 == 0:
                pbar.write(
                    f"Step {global_step} | "
                    f"Loss: {metrics['loss']:.4f} | "
                    f"Res: {metrics['residual_loss']:.4f} | "
                    f"Recon: {metrics['reconst_loss']:.4f} | "
                    f"Pred: {metrics['prediction_loss']:.4f} | "
                    f"Sparsity: {metrics['sparsity_ratio']:.3f}"
                )
            
            # Periodic evaluation and checkpoint saving
            if global_step % 500 == 0 or global_step == cfg.TRAIN.NUM_STEPS - 1:
                # Evaluate on test data
                eval_results = evaluate_finance(
                    model, test_init, test_future, max_horizon=50
                )
                
                logger.log_scalar('eval/mean_mse_reencode', eval_results['mean_mse_reencode'], global_step)
                logger.log_scalar('eval/mean_mse_no_reencode', eval_results['mean_mse_no_reencode'], global_step)
                logger.log_scalar('eval/final_mse_reencode', eval_results['final_mse_reencode'], global_step)
                logger.log_scalar('eval/final_mse_no_reencode', eval_results['final_mse_no_reencode'], global_step)
                
                pbar.write(f"Step {global_step} | Eval MSE (reencode): {eval_results['mean_mse_reencode']:.4f}")
                
                # Compute validation loss
                val_loss = 0.0
                val_batches = 0
                model.eval()
                with torch.no_grad():
                    for val_batch in val_loader:
                        if isinstance(val_batch, (list, tuple)):
                            vx, vnx = val_batch
                            vx, vnx = vx.to(device), vnx.to(device)
                            _, val_metrics = model.loss(vx, vnx)
                        else:
                            vx = val_batch.to(device)
                            _, val_metrics = model.loss_sequence(vx, dt)
                        val_loss += val_metrics['loss']
                        val_batches += 1
                        if val_batches >= 10:  # Limit val batches for speed
                            break
                model.train()
                
                avg_val_loss = val_loss / max(val_batches, 1)
                logger.log_scalar('val/loss', avg_val_loss, global_step)
                
                # Save checkpoint
                checkpoint_dict = {
                    'step': global_step,
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'config': cfg.to_dict(),
                    'metrics': metrics,
                    'finance_metadata': finance_env.metadata,
                }
                
                # Save latest checkpoint
                torch.save(checkpoint_dict, run_dir / 'last.pt')
                
                # Save best checkpoint based on validation loss
                if avg_val_loss < best_eval_error:
                    best_eval_error = avg_val_loss
                    torch.save(checkpoint_dict, run_dir / 'checkpoint.pt')
                    pbar.write(f"  Saved best checkpoint (val loss: {best_eval_error:.4f})")
            
            global_step += 1
        
    pbar.close()
    
    # Save final metrics and close logger
    with open(run_dir / 'final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.close()
    
    # Run finance-specific evaluation
    print("-" * 80)
    print("Running finance evaluation suite...")
    
    # Load best checkpoint for final evaluation
    best_ckpt_path = run_dir / 'checkpoint.pt'
    if best_ckpt_path.exists():
        best_ckpt = torch.load(best_ckpt_path, map_location=device)
        model.load_state_dict(best_ckpt['model_state_dict'])
        print("Loaded best checkpoint for evaluation")
    
    # Final evaluation with longer horizon
    final_eval = evaluate_finance(
        model, test_init, test_future, 
        max_horizon=100,
        periodic_reencode_periods=[5, 10, 25],
    )
    
    # Save evaluation results
    eval_results_path = run_dir / 'evaluation_results.json'
    eval_summary = {
        'mean_mse_reencode': final_eval['mean_mse_reencode'],
        'mean_mse_no_reencode': final_eval['mean_mse_no_reencode'],
        'final_mse_reencode': final_eval['final_mse_reencode'],
        'final_mse_no_reencode': final_eval['final_mse_no_reencode'],
        'mse_reencode_curve': final_eval['mse_reencode'].tolist(),
        'mse_no_reencode_curve': final_eval['mse_no_reencode'].tolist(),
        'all_modes_mean_mse': final_eval.get('mean_mses', {}),
        'best_mode': final_eval.get('best_mode', 'every_step'),
        'best_mse': final_eval.get('best_mse', 0),
    }
    with open(eval_results_path, 'w') as f:
        json.dump(eval_summary, f, indent=2)
    
    print("-" * 80)
    print(f"Training complete! Results saved to {run_dir}")
    
    return model


def get_device(device_arg: str) -> str:
    """Auto-detect the best available device.
    
    Priority order:
    1. Use explicitly requested device if available
    2. MPS (Metal Performance Shaders) on macOS
    3. CUDA on Linux/Windows
    4. CPU as fallback
    
    Args:
        device_arg: Requested device ('cpu', 'cuda', 'mps', or 'auto')
        
    Returns:
        Device string ('cpu', 'cuda', or 'mps')
    """
    # If explicitly CPU, use it
    if device_arg == 'cpu':
        return 'cpu'
    if device_arg == 'mps':
        return 'mps' if torch.backends.mps.is_available() else 'cpu'
    if device_arg == 'cuda':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    
    if torch.cuda.is_available():
        return 'cuda'
    elif torch.backends.mps.is_available():
        return 'mps'
    else:
        return 'cpu'


def main():
    """Command-line interface for training."""
    parser = argparse.ArgumentParser(description='Train Koopman Autoencoder for Finance')
    
    # Configuration
    parser.add_argument('--config', type=str, default='finance_sparse',
                        choices=['finance_sparse'],
                        help='Training configuration preset')
    
    # Training
    parser.add_argument('--num_steps', type=int, default=5000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')
    parser.add_argument('--k_matrix_lr', type=float, default=None,
                        help='K matrix learning rate')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # Model
    parser.add_argument('--target_size', type=int, default=None,
                        help='Latent dimension')
    parser.add_argument('--encoder_layers', type=str, default=None,
                        help='Encoder layers as comma-separated integers (e.g. "1024,1024")')
    parser.add_argument('--sparsity_coeff', type=float, default=None,
                        help='Sparsity loss weight')
    parser.add_argument('--reconst_coeff', type=float, default=None,
                        help='Reconstruction loss weight')
    parser.add_argument('--res_coeff', type=float, default=None,
                        help='Residual loss weight')
    parser.add_argument('--pred_coeff', type=float, default=None,
                        help='Prediction loss weight')
    
    # Training mode
    parser.add_argument('--pairwise', action='store_true',
                        help='Use single-step training')
    parser.add_argument('--sequence_length', type=int, default=None,
                        help='Sequence length for training')
    parser.add_argument('--embedding_dim', type=int, default=None,
                        help='Embedding dimension (number of lagged timesteps in embedding)')
    parser.add_argument('--resample_weekly', action='store_true',
                        help='Resample data to weekly')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./runs/kae_finance',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'mps', 'auto'],
                        help='Device to train on')
    parser.add_argument('--data_path', type=str, default=None,
                        help='Path to data directory')
    
    args = parser.parse_args()
    
    cfg = get_config(args.config)
    
    cfg.TRAIN.NUM_STEPS = args.num_steps
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed
    
    if args.lr is not None:
        cfg.TRAIN.LR = args.lr
        if args.k_matrix_lr is None:
             cfg.TRAIN.K_MATRIX_LR = args.lr * 0.1
    if args.k_matrix_lr is not None:
        cfg.TRAIN.K_MATRIX_LR = args.k_matrix_lr
    if args.target_size is not None:
        cfg.MODEL.TARGET_SIZE = args.target_size
    if args.encoder_layers is not None:
        cfg.MODEL.ENCODER.LAYERS = [int(x) for x in args.encoder_layers.split(',')]
    if args.sparsity_coeff is not None:
        cfg.MODEL.SPARSITY_COEFF = args.sparsity_coeff
    if args.reconst_coeff is not None:
        cfg.MODEL.RECONST_COEFF = args.reconst_coeff
    if args.res_coeff is not None:
        cfg.MODEL.RES_COEFF = args.res_coeff
    if args.pred_coeff is not None:
        cfg.MODEL.PRED_COEFF = args.pred_coeff
    
    if args.pairwise:
        cfg.TRAIN.USE_SEQUENCE_LOSS = False
    if args.sequence_length is not None:
        cfg.TRAIN.SEQUENCE_LENGTH = args.sequence_length
    if args.embedding_dim is not None:
        cfg.ENV.FINANCE.EMBEDDING_DIM = args.embedding_dim
    if args.resample_weekly:
        cfg.ENV.FINANCE.RESAMPLE_WEEKLY = True
    if args.data_path is not None:
        cfg.ENV.FINANCE.CACHE_DIR = args.data_path
    
    device = get_device(args.device)
    print(f"Using device: {device}")
    
    train(cfg, log_dir=args.log_dir, checkpoint_path=args.checkpoint, device=device)


if __name__ == '__main__':
    main()
