#!/bin/bash
#SBATCH --job-name="null-pilot"
#SBATCH --nodelist=piora1,piora2,piora5,piora6
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=8
#SBATCH --output=logs/null_pilot_%A_%a.log
#SBATCH --array=0-2
#
# Null-baseline pilot: 3 architecture-default random-init Mixtral-8x7B DAGs on c4.
# One SLURM array task per seed (0, 1, 2). Each task runs single-node, uses 4 GPUs
# via accelerate's dispatch_model to shard the ~94GB bf16 model.
#
# Output:
#   ${result_path}/circuits/dag_mixtral-8x7b_c4_rand_s{0,1,2}.pt
#
# Submit:
#   sbatch experiments/launch_random_init_pilot.sh
#
# After all three land, run a quick sanity check:
#   python experiments/check_random_init_pilot.py
# (verifies non-degenerate routing, finite act values, etc.)

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

# Preserve the HF cache path.
export HF_HOME="${HF_HOME:-$HOME/.hugging_face}"

# Parallelise CPU init (kaiming_uniform_ / dtype-cast benefit from OpenMP).
# Match --cpus-per-task above so we don't oversubscribe across array tasks
# running on the same node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

SEED=${SLURM_ARRAY_TASK_ID:-0}
MODEL="mixtral-8x7b"
DATASET="c4"
N_PROMPTS=500

echo "[seed=$SEED] $MODEL/$DATASET  host=$(hostname)  t=$(date +%H:%M:%S)"
echo "  GPUs: $(nvidia-smi -L | wc -l) visible"

${ENV_BIN}/python experiments/build_dag.py \
    --model "$MODEL" \
    --dataset "$DATASET" \
    --n_prompts "$N_PROMPTS" \
    --B 16 \
    --random-init \
    --seed "$SEED"

echo "[seed=$SEED] done at $(date +%H:%M:%S)"
