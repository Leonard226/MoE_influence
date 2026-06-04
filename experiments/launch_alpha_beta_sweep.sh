#!/bin/bash
#SBATCH --array=0-575               # 64 sources × 9 target chunks = 576 work units
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=24G           # peak per task ~8-12GB for qwen3-235b/dsv2
                                    # pairs at Q=0.9; 24G gives ~2x headroom and
                                    # lets SLURM pack heterogeneously so cheap
                                    # Mixtral chunks aren't artificially throttled.
#SBATCH --nodelist=piora[5-8]       # restrict to piora5..piora8 (CPU-only sweep)
#SBATCH --output=logs/ab_sweep_%A_%a.log
#
# Pairwise α × Q quantile FGW sweep across all (model, task) tuples at fixed
# β = 0.5. CPU-only (no GPU needed -- FGW is POT/numpy/scipy).
#
# Submit:
#   mkdir -p logs
#   sbatch experiments/launch_alpha_beta_sweep.sh
#
# 64 (model, task) sources × 9 target chunks = 576 array tasks.
# Each task is independent. Per source:
#   - For each Q ∈ {0.9, 0.99, 0.999}: build ONE triple at β = 0.5 with
#     F + mass from the dense graph and C_path from the Q-sparsified graph
#     (vertices isolated under the threshold are dropped).
# Per target in the chunk:
#   - For each Q: build the target triple the same way.
#   - For each α ∈ {0, 0.5, 1}: compute FGW(source, target).
# Writes a slice .npz into
#     {result_path}/circuits/alpha_beta_sweep/sweep_src{SS}_chunk{CC}.npz
# of shape (n_targets_in_chunk, 3, 3) indexed (local_target, α_idx, Q_idx).
#
# Resumable: each task checkpoints after every target; restarting an array
# task picks up where it left off.
#
# After all array tasks finish:
#   python experiments/aggregate_alpha_beta_sweep.py
# stitches the slices into S_full.npz of shape (64, 64, 3, 3) indexed
# (src, tgt, α, Q).

set -euo pipefail

N_SOURCES=64
N_CHUNKS=9

ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
SOURCE_IDX=$(( ARRAY_IDX / N_CHUNKS ))
CHUNK_IDX=$(( ARRAY_IDX % N_CHUNKS ))

echo "[$ARRAY_IDX] source_idx=$SOURCE_IDX  chunk_idx=$CHUNK_IDX"
echo "host=$(hostname)  cpus=${SLURM_CPUS_PER_TASK:-?}"

# Find project root (so SLURM_SUBMIT_DIR can be either repo root or experiments/).
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$PROJECT_ROOT" != "/" ] && [ ! -f "$PROJECT_ROOT/config.yaml" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -f "$PROJECT_ROOT/config.yaml" ]; then
    echo "ERROR: cannot find project root (no config.yaml found walking up from ${SLURM_SUBMIT_DIR:-$PWD})" >&2
    exit 1
fi
cd "$PROJECT_ROOT"
mkdir -p logs

ENV_BIN=/scratch/sleonard/miniconda3/envs/megatron/bin
export PATH="${ENV_BIN}:${PATH}"
export LD_LIBRARY_PATH="/scratch/sleonard/miniconda3/envs/megatron/lib:${LD_LIBRARY_PATH:-}"

# Pin BLAS / OpenMP to 1 thread (matches cpus-per-task=1). FGW is dominated
# by single-threaded code (POT's Frank-Wolfe, scipy Dijkstra), and pinning
# prevents BLAS from spawning extra threads that oversubscribe when many
# array tasks share a node.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

${ENV_BIN}/python experiments/run_alpha_beta_sweep.py \
    --source-idx "$SOURCE_IDX" \
    --target-chunk "$CHUNK_IDX" \
    --num-chunks "$N_CHUNKS"