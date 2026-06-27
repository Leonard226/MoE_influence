#!/bin/bash
#SBATCH --array=0-223
#SBATCH --nodelist=piora3,piora4,piora7,piora8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --job-name="loo-ablate"
#SBATCH --output=logs/loo_ablation_%A_%a.log
#
# Leave-one-out (LOO) feature-ablation sweep at alpha = 0 (Wasserstein-only).
# For each unordered model pair (28) x dataset (8) = 224 pair-tasks, compute
# FGW similarity under 7 feature-ablation conditions x 3 Q values = 21 cells.
#
# One SLURM array task per pair-task (--pair-task-idx). Each task builds its
# 6 base triples (2 models x 3 Q) once and runs 21 FGW calls sequentially.
# Resume-friendly: skips pair-tasks whose output already exists and is full.
#
# Settings match the headline sweep:
#   alpha = 0 (Wasserstein-only; C is unused so structural_mode irrelevant)
#   n_init = 3
#   act_norm  = log_max
#   load_norm = log_max
#
# Parallelism layout:
#   - 224 array tasks. piora3,4,7,8 are the 4 CPU nodes; ~32 cores each,
#     4 cpus-per-task -> ~32 concurrent slots, ~7 waves needed.
#   - OMP/MKL/OPENBLAS = 4 so scipy + POT use all 4 cores per task (same
#     reasoning as the alpha-beta sweep launcher).
#
# piora1,2 are the A100 nodes -- intentionally excluded so the LOO sweep
# doesn't burn GPU node time on a CPU-bound workload.
#
# Submit:
#   sbatch experiments/launch_feature_ablation.sh
#
# Output:
#   ${result_path}/circuits/feature_ablation_logact_logload/
#       S_loo_pair_{000..223}.npz   (one file per pair-task)
#
# After all 224 tasks finish, merge per-pair files:
#   python experiments/feature_ablation_sweep.py --merge \
#       --act-norm log_max --load-norm log_max
#
# Analyze:
#   python experiments/analyze_feature_ablation_breakdown.py \
#       --act-norm log_max --load-norm log_max

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

PAIR_TASK_IDX=${SLURM_ARRAY_TASK_ID:-0}

echo "[$PAIR_TASK_IDX] host=$(hostname)  t=$(date +%H:%M:%S)"

${ENV_BIN}/python experiments/feature_ablation_sweep.py \
    --pair-task-idx "$PAIR_TASK_IDX" \
    --act-norm log_max \
    --load-norm log_max
