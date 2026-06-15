#!/bin/bash
#SBATCH --array=0-63
#SBATCH --nodelist=piora3,piora4
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name="abQ-sweep"
#SBATCH --output=logs/abQ_sweep_%A_%a.log
#
# Headline α × β=0.5 × Q sweep under the new log-max load and log-max act
# normalisations. One SLURM array task per source-idx (64 sources total).
#
# Submit:
#   sbatch experiments/launch_alpha_beta_sweep.sh
#
# Layout on the cluster:
#   - 64 array tasks, distributed across piora3 + piora4.
#   - Each task: --cpus-per-task=4. With 128 cores/node, 32 tasks fit per
#     node = 64 concurrent across the two nodes (one per source-idx).
#   - OMP/MKL/OPENBLAS pinned to 1 thread so co-located tasks don't
#     oversubscribe.
#
# Output:
#   ${result_path}/circuits/alpha_beta_sweep_logact_logload/
#     sweep_src{NN}_chunk00.npz   (one per source)
#
# Resume: re-submit the same array; run_alpha_beta_sweep.py resumes from the
#     existing sweep_src{NN}_chunk00.npz file (per-Q checkpointing).
#
# Aggregate after the array finishes:
#   python experiments/aggregate_alpha_beta_sweep.py \
#       --act-norm log_max --load-norm log_max

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

ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
echo "[$ARRAY_IDX] source-idx=$ARRAY_IDX  host=$(hostname)  t=$(date +%H:%M:%S)"

${ENV_BIN}/python experiments/run_alpha_beta_sweep.py \
    --source-idx "$ARRAY_IDX" \
    --target-chunk 0 \
    --num-chunks 1 \
    --act-norm log_max \
    --load-norm log_max
