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



def train(
    cfg: Config,
    log_dir: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    device: str = 'cuda',
) -> nn.Module:
    """Main training function.
    
    Args:
        cfg: Configuration object
        log_dir: Directory for tensorboard logs and checkpoints
        checkpoint_path: Path to checkpoint to resume from
        device: Device to train on ('cpu', 'cuda', 'mps')
        
    Returns:
        Trained model
    """
    print("Initializing training...")
    
    # Setup logging directory and save config
    if log_dir is None:
        log_dir = './runs/kae'
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
    elif env_name == 'finance':  
        dt = 1.0
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
                                'generic_prediction', 'lista', 'lista_nonlinear'],
                        help='Training configuration preset')
    parser.add_argument('--env', type=str, default='duffing',
                        choices=['duffing', 'pendulum', 'lotka_volterra', 
                                'lorenz63', 'parabolic', 'lyapunov', 'finance'],
                        help='Dynamical system environment')
    
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
    parser.add_argument('--log_dir', type=str, default='./runs/kae',
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

