#!/bin/bash
#SBATCH --job-name="null-small"
#SBATCH --nodelist=piora1,piora2,piora5,piora6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/null_small_%A_%a.log
#SBATCH --array=0-11
#
# Null-baseline DAGs for the four single-node MoE architectures (the easy
# cases). Each (model, seed) is one SLURM array task with 4 GPUs.
#
# Models:
#   olmoe              -- single-GPU class; uses 1 of the 4 GPUs
#   deepseek-v2-lite   -- single-GPU class; uses 1 of the 4 GPUs
#   qwen3-30b-a3b      -- multi_gpu=True; shards across 4 GPUs
#   phi-3.5-moe        -- multi_gpu=True; shards across 4 GPUs
#
# Array layout: ARRAY_IDX = MODEL_IDX * N_SEEDS + SEED
#   0,1,2  -> olmoe seeds 0,1,2
#   3,4,5  -> deepseek-v2-lite seeds 0,1,2
#   6,7,8  -> qwen3-30b-a3b seeds 0,1,2
#   9,10,11 -> phi-3.5-moe seeds 0,1,2
#
# Mixtral-8x7B is already done by the pilot launcher (skipped here).
# Mixtral-8x22B, DeepSeek-V2, Qwen3-235B require the multi-node adaptation
# (deferred).
#
# Submit:
#   sbatch experiments/launch_random_init_small.sh
#
# Output:
#   ${result_path}/circuits/dag_<model>_c4_rand_s{0,1,2}.pt
#
# After all 12 tasks land, sanity-check per model:
#   for M in olmoe deepseek-v2-lite qwen3-30b-a3b phi-3.5-moe; do
#       python experiments/check_random_init_pilot.py --model $M
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

# HF cache.
export HF_HOME="${HF_HOME:-$HOME/.hugging_face}"

# Parallelise CPU init (kaiming_uniform_ / dtype-cast benefit from OpenMP).
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# 4 models x 3 seeds = 12 tasks.
MODELS=(olmoe deepseek-v2-lite qwen3-30b-a3b phi-3.5-moe)
N_SEEDS=3

ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
MODEL_IDX=$(( ARRAY_IDX / N_SEEDS ))
SEED=$(( ARRAY_IDX % N_SEEDS ))
MODEL="${MODELS[$MODEL_IDX]}"
DATASET="c4"
N_PROMPTS=500

echo "[task $ARRAY_IDX] model=$MODEL seed=$SEED  host=$(hostname)  t=$(date +%H:%M:%S)"
echo "  GPUs visible: $(nvidia-smi -L | wc -l)"

${ENV_BIN}/python experiments/build_dag.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --n_prompts "$N_PROMPTS" \
    --B 16 \
    --random-init \
    --seed "$SEED"

echo "[task $ARRAY_IDX] done at $(date +%H:%M:%S)"
