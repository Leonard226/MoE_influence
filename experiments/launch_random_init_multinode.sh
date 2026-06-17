#!/bin/bash
#SBATCH --job-name="null-multi"
#SBATCH --nodelist=piora1,piora2
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/null_multi_%A_%a.log
#SBATCH --array=0-8
#
# Multi-node random-init DAG builds for the three big MoE architectures.
# 3 models x 3 seeds = 9 array tasks. Each task takes 2 GPU nodes (8 GPUs)
# constrained to piora1,piora2; tasks run sequentially (one wave at a time
# because only 2 nodes are available for the 2-node allocations).
#
# Array layout: ARRAY_IDX = MODEL_IDX * N_SEEDS + SEED
#   0,1,2  -> mixtral-8x22b seeds 0,1,2
#   3,4,5  -> deepseek-v2 seeds 0,1,2
#   6,7,8  -> qwen3-235b-a22b seeds 0,1,2
#
# Submit:
#   sbatch experiments/launch_random_init_multinode.sh
#
# Output:
#   ${result_path}/circuits/dag_<model>_c4_rand_s{0,1,2}.pt
#
# After all 9 tasks land:
#   for M in mixtral-8x22b deepseek-v2 qwen3-235b-a22b; do
#       python experiments/check_random_init_pilot.py --model "$M"
#   done

set -euo pipefail

# Find project root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$PROJECT_ROOT" != "/" ] && [ ! -f "$PROJECT_ROOT/config.yaml" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -f "$PROJECT_ROOT/config.yaml" ]; then
    echo "ERROR: cannot find project root walking up from ${SLURM_SUBMIT_DIR:-$PWD}" >&2
    exit 1
fi
cd "$PROJECT_ROOT"
mkdir -p logs

ENV_BIN=/scratch/sleonard/miniconda3/envs/megatron/bin
export PATH="${ENV_BIN}:${PATH}"
export LD_LIBRARY_PATH="/scratch/sleonard/miniconda3/envs/megatron/lib:${LD_LIBRARY_PATH:-}"

# HF cache (for config + tokenizer; weights are NOT pulled).
export HF_HOME="${HF_HOME:-$HOME/.hugging_face}"

# Parallelise CPU-side work.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# Resolve master node IP from this array task's allocation.
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR="${nodes[0]}"
# Unique port per array task to avoid c10d rendezvous collisions when
# multiple tasks run concurrently.
export MASTER_PORT=$(( 29500 + SLURM_ARRAY_TASK_ID ))

# NCCL: prefer InfiniBand, quiet logs unless requested.
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# 3 models x 3 seeds.
MODELS=(mixtral-8x22b deepseek-v2 qwen3-235b-a22b)
N_SEEDS=3
ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
MODEL_IDX=$(( ARRAY_IDX / N_SEEDS ))
SEED=$(( ARRAY_IDX % N_SEEDS ))
MODEL="${MODELS[$MODEL_IDX]}"
DATASET="c4"
N_PROMPTS=500

echo "[task $ARRAY_IDX] model=$MODEL seed=$SEED  nodes=${nodes[*]}  master=${MASTER_ADDR}:${MASTER_PORT}  t=$(date +%H:%M:%S)"

srun --export=ALL ${ENV_BIN}/torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    experiments/build_dag_multinode.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --n_prompts "$N_PROMPTS" \
    --B 4 \
    --random-init \
    --seed "$SEED"

echo "[task $ARRAY_IDX] done at $(date +%H:%M:%S)"
