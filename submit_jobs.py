import os
import itertools
from pathlib import Path

# Hyperparameters to search over
# Modify these lists to run a grid search
search_space = {
    "sequence_length": [2, 5, 10],
    "lr": [1e-2, 1e-3, 1e-4],
    "k_matrix_lr": [None], # None will default to 0.1 * lr
    "res_coeff": [0.1],
    "reconst_coeff": [0.01, 0.1, 0.5, 1.0],
    "pred_coeff": [0.001, 0.01, 0.1],
    "sparsity_coeff": [0.01, 0.1, 0.5, 1.0, 1.5],
    "target_size": [512, 1024, 2048],
    "encoder_layers": ["512,512", "1024,1024"],
}

# Template content
template = """#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --output=/network/scratch/u/lia/slurm-%j.out
#SBATCH --error=/network/scratch/u/lia/slurm-%j.err
#SBATCH --time=2:59:00
#SBATCH --partition=main
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --ntasks=1

# --- Environment & Data ---
module load python/3.10

# Copy dataset to fast local storage ($SLURM_TMPDIR)
# If you have a dataset on /network/datasets, uncomment and set path:
# cp -r /network/datasets/YOUR_DATASET $SLURM_TMPDIR/

# --- Execution ---
echo "Starting job on node $(hostname)"

# Note: data_path is set to $SLURM_TMPDIR. 
# If not copying data there, the finance env might try to download it 
# (which requires internet access) or use the cache in $SLURM_TMPDIR if copied.

uv run python train.py \\
    --config finance_sparse \\
    --env finance \\
    --num_steps 10000 \\
    --sequence_length {sequence_length} \\
    --lr {lr} \\
    {k_matrix_lr_arg}\\
    --res_coeff {res_coeff} \\
    --reconst_coeff {reconst_coeff} \\
    --pred_coeff {pred_coeff} \\
    --sparsity_coeff {sparsity_coeff} \\
    --target_size {target_size} \\
    --encoder_layers "{encoder_layers}" \\
    --data_path $SLURM_TMPDIR

# --- Cleanup/Saving ---
# Copy results back to Scratch if needed
# cp -r runs/kae_finance /network/scratch/u/lia/
"""

def submit_jobs():
    jobs_dir = Path("jobs")
    jobs_dir.mkdir(exist_ok=True)
    
    keys, values = zip(*search_space.items())
    combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    print(f"Generating {len(combinations)} job scripts in {jobs_dir}...")
    
    for i, params in enumerate(combinations):
        # Create a descriptive job name or just an index
        job_name = f"koopman_fin_{i}"

        # Handle optional k_matrix_lr
        k_matrix_lr = params.get("k_matrix_lr")
        if k_matrix_lr is None:
             k_matrix_lr_arg = ""
        else:
             k_matrix_lr_arg = f"--k_matrix_lr {k_matrix_lr} \\"
        
        # Create sbatch content
        content = template.format(
            job_name=job_name,
            k_matrix_lr_arg=k_matrix_lr_arg,
            **{k: v for k, v in params.items() if k != "k_matrix_lr"}
        )
        
        script_path = jobs_dir / f"{job_name}.sbatch"
        with open(script_path, "w") as f:
            f.write(content)
            
        print(f"Created {script_path}")
        
    print(f"\nGenerated {len(combinations)} job scripts.")
    print("To submit all jobs at once, run:")
    print(f"  for f in {jobs_dir}/*.sbatch; do sbatch $f; done")
    
    # Optional: automatic submission
    if len(combinations) > 0:
        response = input("\nDo you want to submit these jobs now? (y/N): ")
        if response.lower() == 'y':
            import subprocess
            for i in range(len(combinations)):
                script_path = jobs_dir / f"koopman_fin_{i}.sbatch"
                print(f"Submitting {script_path}...")
                subprocess.run(["sbatch", str(script_path)])

if __name__ == "__main__":
    submit_jobs()

