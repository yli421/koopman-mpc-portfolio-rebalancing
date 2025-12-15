#!/bin/bash
#SBATCH --job-name=my_uv_job
#SBATCH --output=slurm/slurm-%j.out  # Save output to scratch
#SBATCH --error=slurm/slurm-%j.err

# --- Time Limit ---
# Less than 3 hours places you in a favorable scheduling bin.
#SBATCH --time=2:59:00                                       

# --- Partition ---
# "main" allows up to 5 days, but is preemptible. 
# "unkillable" is an option for < 2 days & 1 GPU, but resources are limited.
#SBATCH --partition=long                                     

# --- Compute Resources ---
# Requesting 1 RTX8000 (Performant 48GB VRAM).
# Optimal ratio for RTX8000 is 1 GPU : 8 CPUs : 48GB RAM.
#SBATCH --gres=gpu:1                                 
#SBATCH --cpus-per-task=8                                    
#SBATCH --mem=32G                                            
#SBATCH --ntasks=1                                           

# --- Environment & Data ---
# Load Python module if needed for uv to bootstrap (or other dependencies)
module load python/3.10                                      

# Copy dataset to fast local storage ($SLURM_TMPDIR) 
# cp /network/datasets/<dataset> $SLURM_TMPDIR/
NUM_STEPS=5000
SEQ_LENGTH=3
EMBEDDING_DIM=6 # set this to 2x the sequence length
TARGET_SIZE=1024
ENCODER_LAYERS="1024,1024"
SPARSITY_COEFF=0.3
RES_COEFF=1.0
RECONST_COEFF=0.1
PRED_COEFF=0.5
LEARNING_RATE=1e-4
# LOG_DIR="./runs/kae_finance_seq${SEQ_LENGTH}"
LOG_DIR="./runs/kae_finance_weekly_seq${SEQ_LENGTH}"

# --- Execution ---
# Using uv as requested. 
# Ensure 'uv' is in your PATH or installed in your user environment.

echo "Starting job on node $(hostname)"
uv run python train.py \
    --config finance_sparse \
    --env finance \
    --num_steps $NUM_STEPS \
    --sequence_length $SEQ_LENGTH \
    --embedding_dim $EMBEDDING_DIM \
    --resample_weekly \
    --target_size $TARGET_SIZE \
    --encoder_layers $ENCODER_LAYERS \
    --sparsity_coeff $SPARSITY_COEFF \
    --res_coeff $RES_COEFF \
    --reconst_coeff $RECONST_COEFF \
    --pred_coeff $PRED_COEFF \
    --lr $LEARNING_RATE \
    --log_dir $LOG_DIR

# 2. Run the experiment to get baselines, backtest results, and portfolio value plots
uv run python run_experiment.py --search_dir $LOG_DIR

# --- Cleanup/Saving ---
# Copy results back to Scratch [cite: 95]
# cp $SLURM_TMPDIR/<results> /network/scratch/u/username/