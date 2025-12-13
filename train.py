"""
Training script for Koopman Autoencoder models.

This script provides a complete training pipeline for learning Koopman operator
representations of dynamical systems using PyTorch.

Usage:
    python train.py --config generic_sparse --env duffing --num_steps 20000

Or use it programmatically:
    from train import train
    cfg = get_config("generic_sparse")
    cfg.ENV.ENV_NAME = "duffing"
    train(cfg, log_dir="./runs/experiment_001")
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, List

print("Loading torch...")
import torch
import torch.nn as nn
print("Torch loaded.")

print("Loading config...")
from config import Config, get_config
print("Config loaded.")

print("Loading data...")
from data import make_env, VectorWrapper, generate_trajectory
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
    Can easily be replaced with wandb later.
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


def evaluate(
    model: nn.Module,
    x: torch.Tensor,
    env_step_fn,
    num_steps: int = 50,
) -> Dict[str, Any]:
    """Quick evaluation helper used during training and unit tests."""
    
    # Lazy import to avoid loading evaluation module at startup
    from evaluation import rollout_every_step_reencode

    model.eval()
    device = next(model.parameters()).device

    with torch.no_grad():
        true_traj = generate_trajectory(env_step_fn, x.cpu(), length=num_steps)
        pred_traj = rollout_every_step_reencode(model, x.to(device), num_steps)

        pred_traj_cpu = pred_traj.cpu()
        step_error = torch.norm(pred_traj_cpu - true_traj, dim=-1).mean(dim=1)

        return {
            "true_trajectory": true_traj,
            "pred_trajectory": pred_traj_cpu,
            "pred_error": step_error,
            "mean_error": step_error.mean().item(),
            "final_error": step_error[-1].item(),
        }


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


def train_finance(
    cfg: Config,
    log_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
) -> nn.Module:
    """Training function for finance data.
    
    Uses PyTorch DataLoader instead of environment stepping since finance
    data is pre-recorded rather than simulated.
    
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
            
            # Print progress
            if global_step % 100 == 0:
                print(f"Step {global_step}/{cfg.TRAIN.NUM_STEPS} (Epoch {epoch}) | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Res: {metrics['residual_loss']:.4f} | "
                      f"Recon: {metrics['reconst_loss']:.4f} | "
                      f"Pred: {metrics['prediction_loss']:.4f} | "
                      f"Sparsity: {metrics['sparsity_ratio']:.3f}")
            
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
                
                print(f"  Eval | MSE (reencode): {eval_results['mean_mse_reencode']:.4f} | "
                      f"MSE (no reencode): {eval_results['mean_mse_no_reencode']:.4f}")
                
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
                print(f"  Val Loss: {avg_val_loss:.4f}")
                
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
                    print(f"  Saved best checkpoint (val loss: {best_eval_error:.4f})")
            
            global_step += 1
        
        # End of epoch summary
        if epoch_batches > 0:
            avg_epoch_loss = epoch_loss / epoch_batches
            print(f"Epoch {epoch} complete | Avg Loss: {avg_epoch_loss:.4f}")
    
    # Save final metrics and close logger
    with open(run_dir / 'final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.close()
    
    # Plot training metrics
    print("-" * 80)
    print("Plotting training metrics...")
    from plot_training_metrics import plot_metrics
    try:
        plot_metrics(
            log_dir=run_dir,
            metrics_to_plot=None,
            save_path=run_dir / 'training_metrics.png'
        )
        print(f"Training metrics plot saved to {run_dir / 'training_metrics.png'}")
    except Exception as e:
        print(f"Warning: Failed to plot training metrics: {e}")
    
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
    
    # Generate forecasting plots
    print("Generating forecasting plots...")
    try:
        _save_finance_plots(
            eval_results=final_eval,
            finance_env=finance_env,
            output_dir=run_dir,
        )
    except Exception as e:
        print(f"Warning: Failed to generate plots: {e}")
        import traceback
        traceback.print_exc()
    
    print("-" * 80)
    print(f"Training complete! Results saved to {run_dir}")
    
    return model


def _save_finance_plots(
    eval_results: Dict[str, Any],
    finance_env: FinanceEnv,
    output_dir: Path,
) -> None:
    """Generate and save finance-specific evaluation plots.
    
    Creates:
    1. MSE vs horizon curves for different rollout modes
    2. Predicted vs actual returns for sample assets
    3. Prediction correlation scatter plot
    4. MSE bar chart comparing all modes
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Color scheme for plots
    colors = {
        'every_step': '#2ecc71',     # Green
        'no_reencode': '#e74c3c',    # Red
        'periodic_5': '#3498db',     # Blue
        'periodic_10': '#9b59b6',    # Purple
        'periodic_25': '#f39c12',    # Orange
    }
    linestyles = {
        'every_step': '-',
        'no_reencode': '--',
        'periodic_5': '-.',
        'periodic_10': ':',
        'periodic_25': '-',
    }
    
    # 1. MSE vs Horizon plot (all modes)
    fig, ax = plt.subplots(figsize=(12, 7))
    
    mse_curves = eval_results.get('mse_curves', {})
    if not mse_curves:
        # Fallback to old format
        mse_curves = {
            'every_step': eval_results['mse_reencode'],
            'no_reencode': eval_results['mse_no_reencode'],
        }
    
    for mode, curve in mse_curves.items():
        curve_np = curve.numpy() if hasattr(curve, 'numpy') else np.array(curve)
        horizons = range(1, len(curve_np) + 1)
        color = colors.get(mode, '#7f8c8d')
        ls = linestyles.get(mode, '-')
        label = mode.replace('_', ' ').title()
        ax.plot(horizons, curve_np, label=label, linewidth=2, color=color, linestyle=ls)
    
    ax.set_xlabel('Prediction Horizon (days)', fontsize=13)
    ax.set_ylabel('Mean Squared Error', fontsize=13)
    ax.set_title('Multi-Step Prediction Error: Koopman Rollout Modes', fontsize=15)
    ax.legend(loc='upper left', fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(list(mse_curves.values())[0]))
    
    # Annotate best mode
    best_mode = eval_results.get('best_mode', 'every_step')
    best_mse = eval_results.get('best_mse', 0)
    ax.annotate(f'Best: {best_mode.replace("_", " ")} (MSE={best_mse:.4f})',
                xy=(0.98, 0.02), xycoords='axes fraction',
                fontsize=11, ha='right', va='bottom',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    fig.tight_layout()
    fig.savefig(output_dir / 'mse_vs_horizon.png', dpi=200)
    plt.close(fig)
    
    # 2. Predicted vs Actual Returns plot (for a few assets)
    pred = eval_results['pred_reencode']  # [horizon, batch, obs_size]
    true = eval_results['true']  # [horizon, batch, obs_size]
    
    n_assets = finance_env.n_assets
    tickers = finance_env.metadata.get('tickers', [f'Asset_{i}' for i in range(n_assets)])
    
    # Extract current returns (first n_assets of each observation)
    pred_np = pred.numpy() if hasattr(pred, 'numpy') else np.array(pred)
    true_np = true.numpy() if hasattr(true, 'numpy') else np.array(true)
    pred_returns = pred_np[:, 0, :n_assets]  # [horizon, n_assets] - first sequence
    true_returns = true_np[:, 0, :n_assets]
    
    # Plot first 4 assets
    n_plot = min(4, n_assets)
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()
    
    horizon_len = min(50, pred_returns.shape[0])  # Show first 50 days
    
    for i, ax in enumerate(axes[:n_plot]):
        ax.plot(range(horizon_len), true_returns[:horizon_len, i], 
                label='Actual', linewidth=1.5, alpha=0.9, color='#2c3e50')
        ax.plot(range(horizon_len), pred_returns[:horizon_len, i], 
                label='Predicted', linewidth=1.5, alpha=0.8, linestyle='--', color='#e74c3c')
        
        ax.set_title(f'{tickers[i]}', fontsize=13, fontweight='bold')
        ax.set_xlabel('Days Ahead', fontsize=11)
        ax.set_ylabel('Standardized Log-Return', fontsize=11)
        ax.legend(loc='upper right', fontsize=10)
        ax.grid(True, alpha=0.3)
    
    fig.suptitle('Predicted vs Actual Returns (Every-step Re-encoding)', fontsize=15, y=1.01)
    fig.tight_layout()
    fig.savefig(output_dir / 'pred_vs_actual_returns.png', dpi=200)
    plt.close(fig)
    
    # 3. Return correlation scatter plot
    fig, ax = plt.subplots(figsize=(9, 9))
    
    # Flatten predictions and actuals for first few assets
    pred_flat = pred_returns[:horizon_len, :n_plot].flatten()
    true_flat = true_returns[:horizon_len, :n_plot].flatten()
    
    ax.scatter(true_flat, pred_flat, alpha=0.4, s=15, c='#3498db', edgecolors='none')
    
    # Add diagonal line
    lims = [min(ax.get_xlim()[0], ax.get_ylim()[0]),
            max(ax.get_xlim()[1], ax.get_ylim()[1])]
    ax.plot(lims, lims, 'r--', alpha=0.75, linewidth=2, label='Perfect Prediction')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    
    # Calculate correlation
    correlation = np.corrcoef(true_flat, pred_flat)[0, 1]
    
    ax.set_xlabel('Actual Returns', fontsize=13)
    ax.set_ylabel('Predicted Returns', fontsize=13)
    ax.set_title(f'Prediction Correlation: ρ = {correlation:.3f}', fontsize=15)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    fig.tight_layout()
    fig.savefig(output_dir / 'pred_correlation.png', dpi=200)
    plt.close(fig)
    
    # 4. MSE Bar Chart comparing all modes
    mean_mses = eval_results.get('mean_mses', {})
    if mean_mses:
        fig, ax = plt.subplots(figsize=(10, 6))
        
        modes = list(mean_mses.keys())
        mses = list(mean_mses.values())
        bar_colors = [colors.get(m, '#7f8c8d') for m in modes]
        
        bars = ax.bar(range(len(modes)), mses, color=bar_colors, edgecolor='black', linewidth=1.2)
        
        # Highlight best mode
        best_idx = modes.index(best_mode) if best_mode in modes else 0
        bars[best_idx].set_edgecolor('#f1c40f')
        bars[best_idx].set_linewidth(3)
        
        ax.set_xticks(range(len(modes)))
        ax.set_xticklabels([m.replace('_', '\n') for m in modes], fontsize=11)
        ax.set_ylabel('Mean MSE', fontsize=13)
        ax.set_title('Prediction Error by Rollout Mode', fontsize=15)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, mse in zip(bars, mses):
            height = bar.get_height()
            ax.annotate(f'{mse:.4f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=10)
        
        fig.tight_layout()
        fig.savefig(output_dir / 'mse_comparison.png', dpi=200)
        plt.close(fig)
        
        print(f"  Saved: mse_vs_horizon.png, pred_vs_actual_returns.png, pred_correlation.png, mse_comparison.png")
    else:
        print(f"  Saved: mse_vs_horizon.png, pred_vs_actual_returns.png, pred_correlation.png")


def train(
    cfg: Config,
    log_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
) -> nn.Module:
    """Main training function.
    
    Routes to appropriate training function based on environment type.
    For finance data, uses DataLoader-based training.
    For dynamical systems, uses environment stepping.
    
    Args:
        cfg: Configuration object
        log_dir: Directory for tensorboard logs and checkpoints
        checkpoint_path: Path to checkpoint to resume from
        device: Device to train on ('cpu', 'cuda', 'mps')
        
    Returns:
        Trained model
    """
    # Route to finance training if ENV_NAME is "finance"
    if cfg.ENV.ENV_NAME.lower() == "finance":
        return train_finance(cfg, log_dir, checkpoint_path, device)
    
    print("Initializing training...")
    
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
    # MPS doesn't have manual_seed, but manual_seed should be sufficient
    
    print("Creating environment...")
    env = make_env(cfg)
    env = VectorWrapper(env, cfg.TRAIN.BATCH_SIZE)
    
    # Get dt from environment config for ODE integration
    env_name = cfg.ENV.ENV_NAME.lower()
    if env_name == 'duffing':
        dt = cfg.ENV.DUFFING.DT
    elif env_name == 'pendulum':
        dt = cfg.ENV.PENDULUM.DT
    elif env_name == 'lotka_volterra':
        dt = cfg.ENV.LOTKA_VOLTERRA.DT
    elif env_name == 'lorenz63':
        dt = cfg.ENV.LORENZ63.DT
    elif env_name == 'parabolic':
        dt = cfg.ENV.PARABOLIC.DT
    elif env_name == 'lyapunov':
        dt = cfg.ENV.LYAPUNOV.DT
    else:
        dt = 0.01  # default fallback
    
    print("Creating model...")
    model = make_model(cfg, env.observation_size)
    model = model.to(device)
    model.dt = dt  # Store dt in model for use in ODE integration
    
    print("Building optimizer...")
    optimizer = build_optimizer(model, cfg)
    
    start_step = 0
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_step = checkpoint.get('step', 0)
        print(f"Resumed from checkpoint at step {start_step}")
    
    # Pre-generate random number generators for data
    # Each batch gets a non-overlapping seed range to avoid collisions
    # Batch i uses seeds: cfg.SEED + i * BATCH_SIZE to cfg.SEED + (i+1) * BATCH_SIZE - 1
    num_batches = cfg.TRAIN.DATA_SIZE // cfg.TRAIN.BATCH_SIZE
    rngs = [torch.Generator().manual_seed(cfg.SEED + i * cfg.TRAIN.BATCH_SIZE) for i in range(num_batches)]
    
    print(f"Training {cfg.MODEL.MODEL_NAME} on {cfg.ENV.ENV_NAME}")
    print(f"Device: {device}")
    print(f"Observation size: {env.observation_size}")
    print(f"Target size: {cfg.MODEL.TARGET_SIZE}")
    print(f"Batch size: {cfg.TRAIN.BATCH_SIZE}")
    print(f"Total steps: {cfg.TRAIN.NUM_STEPS}")
    print(f"Log directory: {run_dir}")
    print("-" * 80)
    
    best_eval_final_error = float('inf')
    
    for step in range(start_step, cfg.TRAIN.NUM_STEPS):
        # Generate batch
        rng = rngs[step % num_batches]
        
        if cfg.TRAIN.USE_SEQUENCE_LOSS:
            # Generate sequence windows
            x_seq = env.generate_sequence_batch(rng, window_length=cfg.TRAIN.SEQUENCE_LENGTH)
            # x_seq has shape [batch_size, seq_len+1, obs_size]
            x_seq = x_seq.to(device)
            nx = None  # Not used for sequence loss
            metrics = train_step(model, optimizer, x_seq, nx, cfg, dt)
        else:
            # Generate single transitions (backward compatibility)
            x = env.reset(rng)
            nx = env.step(x)
            x = x.to(device)
            nx = nx.to(device)
            metrics = train_step(model, optimizer, x, nx, cfg, dt)
        
        logger.log_dict(metrics, step, prefix='train')
        
        if step % 100 == 0:
            if cfg.TRAIN.USE_SEQUENCE_LOSS:
                print(f"Step {step}/{cfg.TRAIN.NUM_STEPS} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Align: {metrics['alignment_loss']:.4f} | "
                      f"Recon: {metrics['reconst_loss']:.4f} | "
                      f"Pred: {metrics['prediction_loss']:.4f} | "
                      f"Sparsity: {metrics['sparsity_ratio']:.3f}")
            else:
                print(f"Step {step}/{cfg.TRAIN.NUM_STEPS} | "
                      f"Loss: {metrics['loss']:.4f} | "
                      f"Res: {metrics['residual_loss']:.4f} | "
                      f"Recon: {metrics['reconst_loss']:.4f} | "
                      f"Sparsity: {metrics['sparsity_ratio']:.3f}")
        
        # Periodic evaluation and checkpoint saving
        if step % 500 == 0 or step == cfg.TRAIN.NUM_STEPS - 1:
            # Get initial states for evaluation
            if cfg.TRAIN.USE_SEQUENCE_LOSS:
                eval_x = x_seq[:4, 0, :]  # First timestep of first 4 sequences
            else:
                eval_x = x[:4]
            
            eval_results = evaluate(model, eval_x, lambda s: env.step(s), num_steps=200)
            logger.log_scalar('eval/mean_error', eval_results['mean_error'], step)
            logger.log_scalar('eval/final_error', eval_results['final_error'], step)
            
            print(f"  Eval | Mean error: {eval_results['mean_error']:.4f} | "
                  f"Final error: {eval_results['final_error']:.4f}")
            
            # Save checkpoint
            checkpoint_dict = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': cfg.to_dict(),
                'metrics': metrics,
            }
            
            # Save latest checkpoint
            torch.save(checkpoint_dict, run_dir / 'last.pt')
            
            # Save best checkpoint if eval error improved
            if eval_results['final_error'] < best_eval_final_error:
                best_eval_final_error = eval_results['final_error']
                torch.save(checkpoint_dict, run_dir / 'checkpoint.pt')
                print(f"  Saved best checkpoint (final eval error: {best_eval_final_error:.4f})")
    
    # Save final metrics and close logger
    with open(run_dir / 'final_metrics.json', 'w') as f:
        json.dump(metrics, f, indent=2)
    
    logger.close()

    # Plot training metrics
    print("-" * 80)
    print("Plotting training metrics...")
    from plot_training_metrics import plot_metrics
    try:
        plot_metrics(
            log_dir=run_dir,
            metrics_to_plot=None,  # Plot all metrics
            save_path=run_dir / 'training_metrics.png'
        )
        print(f"Training metrics plot saved to {run_dir / 'training_metrics.png'}")
    except Exception as e:
        print(f"Warning: Failed to plot training metrics: {e}")
        print("Continuing with evaluation...")

    print("-" * 80)
    print("Running standardized evaluation suite...")
    print("Loading evaluation module...")
    from evaluation import EvaluationSettings, evaluate_model
    
    def evaluate_checkpoint(checkpoint_path: Path, checkpoint_name: str):
        """Load a checkpoint and evaluate it."""
        if not checkpoint_path.exists():
            print(f"  Skipping {checkpoint_name}: checkpoint not found at {checkpoint_path}")
            return None
        
        print(f"\nEvaluating {checkpoint_name} checkpoint...", flush=True)
        checkpoint = torch.load(checkpoint_path, map_location=device)
        ckpt_step = checkpoint.get('step', 'unknown')
        print(f"  Loaded checkpoint (step={ckpt_step}). Building eval env/model...", flush=True)
        
        # Load model from checkpoint (use unwrapped env for observation_size)
        eval_env = make_env(cfg)
        eval_model = make_model(cfg, eval_env.observation_size)
        eval_model.load_state_dict(checkpoint['model_state_dict'])
        eval_model = eval_model.to(device)
        eval_model.eval()
        eval_model.dt = dt
        
        # Create evaluation settings
        eval_settings = EvaluationSettings()
        eval_settings.systems = [cfg.ENV.ENV_NAME]
        
        # Evaluate
        eval_dir = run_dir / f"evaluation_{checkpoint_name}"
        print(f"  Calling evaluate_model() for systems={eval_settings.systems} ...", flush=True)
        eval_results = evaluate_model(
            model=eval_model,
            cfg=cfg,
            device=device,
            settings=eval_settings,
            output_dir=eval_dir,
        )
        print(f"  evaluate_model() finished for {checkpoint_name}.", flush=True)
        
        # Save results
        results_file = run_dir / f"evaluation_results_{checkpoint_name}.json"
        with open(results_file, "w") as f:
            json.dump(eval_results, f, indent=2)
        
        # Print summary
        primary_system = cfg.ENV.ENV_NAME
        primary_metrics = eval_results.get(primary_system)
        if primary_metrics is not None:
            print(f"  {checkpoint_name.upper()} - Primary system ({primary_system}) MSE summary:")
            for horizon in eval_settings.horizons:
                if primary_system == "parabolic" and horizon > 100:
                    continue
                horizon_key = str(horizon)
                no_re = primary_metrics["modes"]["no_reencode"]["horizons"].get(horizon_key)
                every = primary_metrics["modes"]["every_step"]["horizons"].get(horizon_key)
                best = primary_metrics["best_periodic"].get(horizon_key)
                if no_re is None or every is None:
                    continue
                best_str = "best-PR=N/A" if best is None else f"best-PR={best['mean']:.4e} ({best['mode']})"
                print(
                    f"    Horizon {horizon}: "
                    f"no-reencode={no_re['mean']:.4e}, "
                    f"every-step={every['mean']:.4e}, "
                    f"{best_str}"
                )
        
        print(f"  Evaluation artifacts saved to {eval_dir}")
        return eval_results
    
    # Evaluate both checkpoints
    last_checkpoint = run_dir / 'last.pt'
    best_checkpoint = run_dir / 'checkpoint.pt'
    
    eval_results_last = evaluate_checkpoint(last_checkpoint, "last")
    eval_results_best = evaluate_checkpoint(best_checkpoint, "best")
    
    # Also save a combined summary
    if eval_results_last is not None or eval_results_best is not None:
        summary = {
            "last_checkpoint": eval_results_last is not None,
            "best_checkpoint": eval_results_best is not None,
        }
        summary_file = run_dir / "evaluation_summary.json"
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
    
    print("-" * 80)
    print(f"Training complete! Checkpoints saved to {run_dir}")
    
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
    
    # If explicitly MPS, check availability
    if device_arg == 'mps':
        if torch.backends.mps.is_available():
            return 'mps'
        else:
            print("MPS not available, falling back to CPU")
            return 'cpu'
    
    # If explicitly CUDA, check availability
    if device_arg == 'cuda':
        if torch.cuda.is_available():
            return 'cuda'
        else:
            print("CUDA not available, falling back to CPU")
            return 'cpu'
    
    # Auto-detect: prefer MPS on macOS, then CUDA, then CPU
    if device_arg == 'auto' or device_arg == 'cuda':
        # Check MPS first (macOS)
        if torch.backends.mps.is_available():
            return 'mps'
        # Then CUDA (Linux/Windows with GPU)
        elif torch.cuda.is_available():
            return 'cuda'
        # Fallback to CPU
        else:
            return 'cpu'
    
    return device_arg


def main():
    """Command-line interface for training."""
    print("Starting train.py...")
    parser = argparse.ArgumentParser(description='Train Koopman Autoencoder')
    
    # Configuration
    parser.add_argument('--config', type=str, default='generic',
                        choices=['default', 'generic', 'generic_sparse', 
                                'generic_prediction', 'lista', 'lista_nonlinear',
                                'finance_sparse'],
                        help='Training configuration preset')
    parser.add_argument('--env', type=str, default='duffing',
                        choices=['duffing', 'pendulum', 'lotka_volterra', 
                                'lorenz63', 'parabolic', 'lyapunov', 'finance'],
                        help='Dynamical system environment (use "finance" for portfolio data)')
    
    # Training
    parser.add_argument('--num_steps', type=int, default=20000,
                        help='Number of training steps')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate (overrides config default)')
    parser.add_argument('--seed', type=int, default=0,
                        help='Random seed')
    
    # Model
    parser.add_argument('--target_size', type=int, default=None,
                        help='Latent dimension (overrides config default)')
    parser.add_argument('--sparsity_coeff', type=float, default=None,
                        help='Sparsity loss weight (overrides config default)')
    parser.add_argument('--reconst_coeff', type=float, default=None,
                        help='Reconstruction loss weight (overrides config default)')
    parser.add_argument('--pred_coeff', type=float, default=None,
                        help='Prediction loss weight (overrides config default)')
    parser.add_argument('--lista_alpha', type=float, default=None,
                        help='LISTA soft-threshold alpha (overrides config default)')
    
    # Training mode
    parser.add_argument('--pairwise', action='store_true',
                        help='Use pairwise (single-step) training instead of sequence training')
    parser.add_argument('--sequence_length', type=int, default=10,
                        help='Sequence length for sequence training (overrides config default)')
    
    # Logging
    parser.add_argument('--log_dir', type=str, default='./runs/kae_finance',
                        help='Directory for logs and checkpoints')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to checkpoint to resume from')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['cpu', 'cuda', 'mps', 'auto'],
                        help='Device to train on (auto: auto-detect best available)')
    
    args = parser.parse_args()
    
    # Create config
    cfg = get_config(args.config)
    
    # For finance_sparse config, don't override ENV_NAME unless explicitly specified
    # (finance_sparse already sets ENV_NAME to "finance")
    if args.config == 'finance_sparse':
        # Only override if user explicitly passed --env finance (or other)
        # By default, keep the config's ENV_NAME
        pass
    else:
        cfg.ENV.ENV_NAME = args.env
    
    cfg.TRAIN.NUM_STEPS = args.num_steps
    cfg.TRAIN.BATCH_SIZE = args.batch_size
    cfg.SEED = args.seed
    
    # Override config with command-line args
    if args.lr is not None:
        cfg.TRAIN.LR = args.lr
    if args.target_size is not None:
        cfg.MODEL.TARGET_SIZE = args.target_size
    if args.sparsity_coeff is not None:
        cfg.MODEL.SPARSITY_COEFF = args.sparsity_coeff
    if args.reconst_coeff is not None:
        cfg.MODEL.RECONST_COEFF = args.reconst_coeff
    if args.pred_coeff is not None:
        cfg.MODEL.PRED_COEFF = args.pred_coeff
    if args.lista_alpha is not None:
        cfg.MODEL.ENCODER.LISTA.ALPHA = args.lista_alpha
    
    # Training mode
    if args.pairwise:
        cfg.TRAIN.USE_SEQUENCE_LOSS = False
        print("Using pairwise (single-step) training mode")
    if args.sequence_length is not None:
        cfg.TRAIN.SEQUENCE_LENGTH = args.sequence_length
    
    # Auto-detect device
    device = get_device(args.device)
    print(f"Using device: {device}")
    if device == 'cuda' and torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    elif device == 'mps':
        print("  Using Metal Performance Shaders (MPS)")
    else:
        print("  Using CPU")
    
    # Train
    train(cfg, log_dir=args.log_dir, checkpoint_path=args.checkpoint, device=device)


if __name__ == '__main__':
    main()

