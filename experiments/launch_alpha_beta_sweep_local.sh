#!/bin/bash
#SBATCH --array=0-127
#SBATCH --nodelist=piora5,piora6,piora7,piora8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name="abQ-local"
#SBATCH --output=logs/abQ_local_%A_%a.log
#
# α × β=0.5 × Q sweep under log-max load + log-max act normalisations, with
# structural cost C_local (direct-edge only, no path aggregation). Mirrors
# launch_alpha_beta_sweep.sh exactly except for --structural-mode local.
#
# Motivation: under the local-influence semantics of W_softmax (main.tex §3),
# multi-hop edge composition (shortest-path, hitting time, resistance, Katz)
# has no defensible interpretation. C_local uses only the direct edge magnitude
# as the pairwise structural cost.
#
# Parallelism layout:
#   - 64 sources × 2 target chunks = 128 SLURM array tasks.
#   - 4 nodes (piora5, piora6, piora7, piora8) × 128 cores / 4 cpus-per-task
#     = 128 concurrent slots → all 128 tasks run in a single wave.
#   - OMP/MKL/OPENBLAS pinned to 1 thread (POT/scipy is single-threaded).
#
# Submit:
#   sbatch experiments/launch_alpha_beta_sweep_local.sh
#
# Output:
#   ${result_path}/circuits/alpha_beta_sweep_logact_logload_local/
#     sweep_src{SS}_chunk{CC}.npz   (128 slices, two per source)
#
# Aggregate after the array finishes:
#   python experiments/aggregate_alpha_beta_sweep.py \
#       --act-norm log_max --load-norm log_max --structural-mode local

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

# Pin BLAS threads to 1 so 32 co-located tasks per node don't oversubscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

NUM_CHUNKS=2
ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
SOURCE_IDX=$(( ARRAY_IDX / NUM_CHUNKS ))
TARGET_CHUNK=$(( ARRAY_IDX % NUM_CHUNKS ))

echo "[$ARRAY_IDX] source-idx=$SOURCE_IDX  target-chunk=$TARGET_CHUNK/$((NUM_CHUNKS-1))  host=$(hostname)  t=$(date +%H:%M:%S)"

${ENV_BIN}/python experiments/run_alpha_beta_sweep.py \
    --source-idx "$SOURCE_IDX" \
    --target-chunk "$TARGET_CHUNK" \
    --num-chunks "$NUM_CHUNKS" \
    --act-norm log_max \
    --load-norm log_max \
    --structural-mode local
