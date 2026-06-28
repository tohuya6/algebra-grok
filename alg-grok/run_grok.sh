#!/bin/bash
# SLURM launcher for the few-shot-grokking runs on the Bau H100 cluster.
# Modeled on icl/pre_training/run_icl.sh. One config per job; override via env vars
# (FIXED_P / WD / NAME) so the same script covers the grok run, the control, and the sweep.
#
#   cd alg-grok                                  # submit from here
#   sbatch run_grok.sh                           # grok run   (fixed_p=1, wd=2)
#   WD=0   NAME=ctrl-fp1-wd0 sbatch run_grok.sh   # control    (no weight decay)
#   for p in 0 0.25 0.5 0.75 1.0; do FIXED_P=$p NAME=sweep-fp$p sbatch run_grok.sh; done   # family sweep
#
#SBATCH --job-name=alg-grok
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gpus=h100:1
#SBATCH --time=12:00:00
#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err

set -euo pipefail

# --- environment: EDIT for your cluster -------------------------------------
# Needs: torch (CUDA build), sympy.  nnsight is optional (import is guarded).
source ~/.bashrc
# conda activate <YOUR_ENV>          # <-- set your env here
# module load cuda                   # <-- if your cluster requires it
# ----------------------------------------------------------------------------

cd "$SLURM_SUBMIT_DIR"               # submit from the alg-grok/ directory
mkdir -p out err

FIXED_P=${FIXED_P:-1.0}
WD=${WD:-2.0}
NAME=${NAME:-grok-fp${FIXED_P}-wd${WD}}

# Faithful run: He-style lr/warmup/steps, bf16 on the H100 (~2x + half the memory).
python experiments/train_fixed_p.py \
    --device cuda --bf16 \
    --fixed_p "$FIXED_P" --weight_decay "$WD" \
    --lr 1.5e-4 --lr_warmup_steps 10000 --n_steps 200000 --evaluation_steps 500 \
    --d_model 256 --n_layers 4 --n_heads 4 \
    --num_symbols 12 --max_order 6 --k_shots 32 --batch_size 256 \
    --name "$NAME"
