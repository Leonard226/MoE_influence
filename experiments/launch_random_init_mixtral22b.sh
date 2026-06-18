#!/bin/bash
#SBATCH --job-name="null-m22b"
#SBATCH --nodelist=piora1,piora2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/null_m22b_%A_%a.log
#SBATCH --array=0-2
#
# Null-baseline DAGs for Mixtral-8x22B (single-node, 4-GPU via dispatch_model).
# 3 seeds = 3 array tasks. Uses the SAME path the trained Mixtral-22B DAG was
# built on (build_dag.py + multi_gpu=True), bypassing the multi-node script
# which doesn't have Mixtral in its MODELS registry.
#
# Submit:
#   sbatch experiments/launch_random_init_mixtral22b.sh
#
# Output:
#   ${result_path}/circuits/dag_mixtral-8x22b_c4_rand_s{0,1,2}.pt
#
# After all 3 tasks finish, sanity check:
#   python experiments/check_random_init_pilot.py --model mixtral-8x22b

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

# HF cache (config + tokenizer only; weights are NOT pulled).
export HF_HOME="${HF_HOME:-$HOME/.hugging_face}"

# Parallelise CPU init.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SEED=${SLURM_ARRAY_TASK_ID:-0}
MODEL="mixtral-8x22b"
DATASET="c4"
N_PROMPTS=500

echo "[seed=$SEED] $MODEL/$DATASET  host=$(hostname)  t=$(date +%H:%M:%S)"
echo "  GPUs visible: $(nvidia-smi -L | wc -l)"

${ENV_BIN}/python experiments/build_dag.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --n_prompts "$N_PROMPTS" \
    --B 4 \
    --random-init \
    --seed "$SEED"

echo "[seed=$SEED] done at $(date +%H:%M:%S)"
