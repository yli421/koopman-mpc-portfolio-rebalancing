#!/bin/bash
#SBATCH --job-name=my_uv_job
#SBATCH --output=/network/scratch/u/lia/slurm-%j.out  # Save output to scratch
#SBATCH --error=/network/scratch/u/lia/slurm-%j.err

# --- Time Limit ---
# Less than 3 hours places you in a favorable scheduling bin.
#SBATCH --time=2:59:00                                       

# --- Partition ---
# "main" allows up to 5 days, but is preemptible. 
# "unkillable" is an option for < 2 days & 1 GPU, but resources are limited.
#SBATCH --partition=main                                     

# --- Compute Resources ---
# Requesting 1 RTX8000 (Performant 48GB VRAM).
# Optimal ratio for RTX8000 is 1 GPU : 8 CPUs : 48GB RAM.
#SBATCH --gres=gpu:rtx8000:1                                 
#SBATCH --cpus-per-task=8                                    
#SBATCH --mem=32G                                            
#SBATCH --ntasks=1                                           

# --- Environment & Data ---
# Load Python module if needed for uv to bootstrap (or other dependencies)
module load python/3.10                                      

# Copy dataset to fast local storage ($SLURM_TMPDIR) 
# cp /network/datasets/<dataset> $SLURM_TMPDIR/

# --- Execution ---
# Using uv as requested. 
# Ensure 'uv' is in your PATH or installed in your user environment.
echo "Starting job on node $(hostname)"
uv run python main.py --data_path $SLURM_TMPDIR            # [cite: 94]

# --- Cleanup/Saving ---
# Copy results back to Scratch [cite: 95]
# cp $SLURM_TMPDIR/<results> /network/scratch/u/username/