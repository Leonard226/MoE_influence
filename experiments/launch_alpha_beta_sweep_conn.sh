#!/bin/bash
#SBATCH --array=0-127
#SBATCH --nodelist=piora5,piora6,piora7,piora8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name="abQ-conn"
#SBATCH --output=logs/abQ_conn_%A_%a.log
#
# α × β=0.5 × Q sweep under log-max load + log-max act normalisations, with
# structural cost C_conn (Katz/Neumann path-sum). Mirrors
# launch_alpha_beta_sweep_local.sh except for --structural-mode conn.
#
# Motivation: C_path discards all but the shortest route; C_local discards
# everything except direct edges; neither rewards path redundancy. C_conn
# captures all three of (i) logical distance, (ii) number of paths, (iii)
# cost of paths via the Katz/Neumann path-polynomial:
#   Phi(u, v) = sum over forward paths of prod |W_e| = ((I - W_sparse)^-1)_uv
#   C_conn(u, v) = -log(max(Phi, eps)) / -log(eps)   (clipped to [0, 1])
# We use this strictly as a topological descriptor (no influence-flow claim);
# semantic interpretation lives in the per-vertex features F.
#
# Parallelism layout (identical to the local-mode sweep):
#   - 64 sources × 2 target chunks = 128 SLURM array tasks.
#   - 4 nodes (piora5, piora6, piora7, piora8) × 128 cores / 4 cpus-per-task
#     = 128 concurrent slots → all 128 tasks run in a single wave.
#   - OMP/MKL/OPENBLAS pinned to 1 thread (POT/scipy is single-threaded).
#
# Submit:
#   sbatch experiments/launch_alpha_beta_sweep_conn.sh
#
# Output (NEW DIRECTORY, does not touch existing sweeps):
#   ${result_path}/circuits/alpha_beta_sweep_logact_logload_conn/
#     sweep_src{SS}_chunk{CC}.npz   (128 slices, two per source)
#
# Aggregate after the array finishes:
#   python experiments/aggregate_alpha_beta_sweep.py \
#       --act-norm log_max --load-norm log_max --structural-mode conn
#
# Analyze (writes analysis/with_act/summary.txt + heatmaps in the same dir):
#   python experiments/analyze_alpha_beta_sweep.py \
#       --act-norm log_max --load-norm log_max --structural-mode conn

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
    --structural-mode conn
