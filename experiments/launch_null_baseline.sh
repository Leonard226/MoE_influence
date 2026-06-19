#!/bin/bash
#SBATCH --job-name="null-fgw"
#SBATCH --nodelist=piora5,piora6,piora7,piora8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --output=logs/null_baseline_%A_%a.log
#SBATCH --array=0-2
#
# Null-baseline FGW sweep on c4. One SLURM array task per quantile value
# Q in {0.9, 0.99, 0.999}; each task builds its 32 triples (8 archs x
# [trained + 3 seeds]) once and runs all 1404 FGW calls for that Q in a
# fork-based worker pool.
#
# Parallelism layout:
#   - 3 array tasks (one per Q), each on its own node from piora5-8.
#   - cpus-per-task=16 -> 16 FGW workers per Q-task; triples are forked
#     copy-on-write from the parent so worker memory overhead is minimal.
#   - BLAS pinned to 1 thread per worker (POT/scipy is single-threaded).
#
# Submit:
#   sbatch experiments/launch_null_baseline.sh
#
# Output (per task):
#   ${result_path}/circuits/null_baseline/null_S_Q{q}.csv
#
# After all 3 tasks finish, concatenate:
#   python experiments/run_null_baseline.py --merge

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

# Pin BLAS threads to 1 per worker so the 16 fork workers don't oversubscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

Q_IDX=${SLURM_ARRAY_TASK_ID:-0}

echo "[Q_idx=$Q_IDX] host=$(hostname)  cpus=$SLURM_CPUS_PER_TASK  t=$(date +%H:%M:%S)"

${ENV_BIN}/python experiments/run_null_baseline.py --q-idx "$Q_IDX"

echo "[Q_idx=$Q_IDX] done at $(date +%H:%M:%S)"
