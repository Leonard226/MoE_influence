#!/bin/bash
#SBATCH --array=0-127
#SBATCH --nodelist=piora5,piora6,piora7,piora8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name="abQ-c-b0-g05"
#SBATCH --output=logs/abQ_conn_b0_g05_%A_%a.log
#
# α × Q sweep under log-max load + log-max act normalisations, with:
#   - structural-mode = conn  (Katz path-sum)
#   - beta = 0                (hardcoded in fgw.py: C = C_struct directly,
#                              depth lives only in F via the Wasserstein
#                              channel; not double-encoded into C)
#   - gamma = 0.5             (Katz per-hop discount; damps the combinatorial
#                              dominance of long paths in dense graphs)
#
# Motivation: previous sweeps mixed depth into BOTH the feature matrix F AND
# the structural cost C as |Δdepth|. This double-encoding (a) is redundant and
# (b) conflates depth and connectivity in the GW scalar, causing pairs with
# very different (depth, conn) profiles to score the same. Removing depth from
# C unjams the GW signal; with gamma<1, the connectivity term is also less
# dominated by combinatorial long-path counts.
#
# Note: the filename retains '_b0_' for historical continuity; β has been
# retired as a flag (always 0 now) so it no longer appears in the output dir
# suffix.
#
# Parallelism layout (same as the local-mode sweep):
#   - 64 sources × 2 target chunks = 128 SLURM array tasks.
#   - 4 nodes (piora5, piora6, piora7, piora8) × ~32 cores / 4 cpus-per-task
#     = ~32 concurrent slots → several waves needed.
#   - OMP/MKL/OPENBLAS = 4 to let scipy.linalg (Katz triangular solve) and
#     POT (BCG matrix multiplies) use all 4 cores per task. Each task is a
#     single Python process with no fork pool, so all 4 cores go to BLAS.
#
# Submit:
#   sbatch experiments/launch_alpha_beta_sweep_conn_b0_g05.sh
#
# Output:
#   ${result_path}/circuits/alpha_beta_sweep_logact_logload_conn_g0.5/
#     sweep_src{SS}_chunk{CC}.npz   (128 slices)
#
# Aggregate after the array finishes:
#   python experiments/aggregate_alpha_beta_sweep.py \
#       --act-norm log_max --load-norm log_max \
#       --structural-mode conn --gamma 0.5
#
# Analyze:
#   python experiments/analyze_alpha_beta_sweep.py \
#       --act-norm log_max --load-norm log_max \
#       --structural-mode conn --gamma 0.5

set -euo pipefail

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

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4

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
    --structural-mode conn \
    --gamma 0.5
