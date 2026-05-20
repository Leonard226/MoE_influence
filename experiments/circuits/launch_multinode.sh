#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=4
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --job-name=qwen3-235b-dag
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# Multi-node launch for build_dag_multinode.py (Qwen3-235B-A22B).
# 2 nodes x 4 A100-80GB = 8 ranks; pipeline-parallel over the 94 decoder layers.

set -euo pipefail

mkdir -p logs

# Resolve master node IP from SLURM env.
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT=29500
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  nodes=${nodes[*]}"

# NCCL — favor IB; suppress noisy debug unless explicitly enabled.
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
# Per-rank visibility comes from SLURM/torchrun; do not set CUDA_VISIBLE_DEVICES here.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

srun --kill-on-bad-exit=1 \
    torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "$SCRIPT_DIR/build_dag_multinode.py" \
    --model qwen3-235b-a22b \
    --dataset c4 \
    --n_prompts 5000 \
    --B 4
