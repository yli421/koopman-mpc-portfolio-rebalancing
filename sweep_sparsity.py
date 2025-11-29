import subprocess
import json
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys

def run_sweep():
    # Sweep values (log scale)
    sparsity_coeffs = [0.0, 1e-5, 5e-5, 1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 0.1, 0.2, 0.3, 0.5]
    
    results = []
    
    base_cmd = [
        "uv", "run", "python", "train.py",
        "--config", "generic_sparse",
        "--env", "lyapunov",
        "--num_steps", "20000",
        "--batch_size", "256",
        "--target_size", "64",
        "--reconst_coeff", "0.02",
        "--pred_coeff", "1.0",
        "--pairwise",
        "--seed", "0",
        "--device", "cuda"
    ]
    
    print(f"Starting sweep over sparsity coefficients: {sparsity_coeffs}")
    
    for coeff in sparsity_coeffs:
        print(f"\n{'='*50}")
        print(f"Running with sparsity_coeff = {coeff}")
        print(f"{'='*50}")
        
        cmd = base_cmd + ["--sparsity_coeff", str(coeff)]
        
        # Run the command and capture output
        try:
            process = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            # Extract log directory from output
            # Looking for line: "Log directory: runs/kae/..."
            output = process.stdout
            match = re.search(r"Log directory: (.*)", output)
            
            if match:
                log_dir = Path(match.group(1).strip())
                print(f"Run completed. Log dir: {log_dir}")
                
                # Read evaluation results
                # We want the best checkpoint's evaluation
                eval_file = log_dir / "evaluation_results_best.json"
                
                if eval_file.exists():
                    with open(eval_file, 'r') as f:
                        eval_data = json.load(f)
                    
                    # Extract relevant metric (e.g., mean error for the primary system)
                    # Structure: { "lyapunov": { "best_periodic": { "200": { "mean": ... } } } }
                    # Or check the structure from train.py
                    
                    # Let's look at train.py's evaluate_checkpoint function again or the output structure
                    # It saves what evaluate_model returns.
                    
                    # Based on train.py:
                    # primary_metrics = eval_results.get(primary_system)
                    # ... primary_metrics["modes"]["no_reencode"]["horizons"].get(horizon_key)
                    
                    # We'll extract the mean error for the longest horizon available
                    system_name = "lyapunov"
                    if system_name in eval_data:
                        metrics = eval_data[system_name]
                        # Get the largest horizon
                        horizons = sorted([int(h) for h in metrics["modes"]["no_reencode"]["horizons"].keys()])
                        max_horizon = str(horizons[-1])
                        
                        error = metrics["modes"]["no_reencode"]["horizons"][max_horizon]["mean"]
                        
                        # Also get sparsity ratio if available (it's in final_metrics.json)
                        sparsity_ratio = 0.0
                        final_metrics_path = log_dir / "final_metrics.json"
                        if final_metrics_path.exists():
                            with open(final_metrics_path, 'r') as f:
                                final_metrics = json.load(f)
                                sparsity_ratio = final_metrics.get("sparsity_ratio", 0.0)
                        
                        results.append({
                            "sparsity_coeff": coeff,
                            "error": error,
                            "sparsity_ratio": sparsity_ratio,
                            "log_dir": str(log_dir)
                        })
                        print(f"Result: Error={error:.4e}, Sparsity Ratio={sparsity_ratio:.4f}")
                    else:
                        print(f"Warning: Could not find system '{system_name}' in evaluation results.")
                else:
                    print(f"Warning: Evaluation file {eval_file} not found.")
            else:
                print("Warning: Could not parse log directory from output.")
                print(output[-500:]) # Print last 500 chars
                
        except subprocess.CalledProcessError as e:
            print(f"Error running training with coeff {coeff}:")
            print(e.stderr)
            
    # Plotting
    if results:
        coeffs = [r["sparsity_coeff"] for r in results]
        errors = [r["error"] for r in results]
        sparsity_ratios = [r["sparsity_ratio"] for r in results]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color = 'tab:red'
        ax1.set_xlabel('Sparsity Coefficient (log scale)')
        ax1.set_ylabel('Reconstruction Error (MSE)', color=color)
        ax1.plot(coeffs, errors, marker='o', color=color, label='Error')
        ax1.tick_params(axis='y', labelcolor=color)
        ax1.set_xscale('symlog', linthresh=1e-5)
        
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('Sparsity Ratio (% zeros)', color=color)
        ax2.plot(coeffs, sparsity_ratios, marker='s', color=color, linestyle='--', label='Sparsity')
        ax2.tick_params(axis='y', labelcolor=color)
        
        plt.title('Parameter Sweep: Sparsity Coefficient')
        fig.tight_layout()
        plt.grid(True, which="both", ls="-", alpha=0.2)
        
        plot_path = "sparsity_sweep_results.png"
        plt.savefig(plot_path)
        print(f"\nSweep complete. Plot saved to {plot_path}")
        
        # Print best
        best_run = min(results, key=lambda x: x["error"])
        print(f"\nBest Sparsity Coefficient: {best_run['sparsity_coeff']} (Error: {best_run['error']:.4e})")
        
        # Save raw results
        with open("sparsity_sweep_results.json", "w") as f:
            json.dump(results, f, indent=2)
            
    else:
        print("No results collected.")

if __name__ == "__main__":
    run_sweep()

