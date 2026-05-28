#!/bin/bash
#SBATCH --array=0-63                # 8 models × 8 datasets = 64 array tasks
#SBATCH --cpus-per-task=1
#SBATCH --output=logs/classify_%A_%a.log
#
# Parallel build of token classifications: one SLURM array task per
# (model, dataset) pair. CPU-only (no GPU). Submit:
#
#   sbatch experiments/launch_classifications.sh
#
# Resumable: each task calls build_classifications.py with --models X
# --datasets Y, which internally skips if the cache pickle already exists.
# So already-built pairs (the ~27 you've done so far) are no-ops.
#
# After the array finishes, every classify_<model>_<task>.pkl will exist
# under {result_path}/circuits/classifications/. Then submit the sweep.

set -euo pipefail

# Find project root.
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

# spaCy + HF tokenizer don't benefit from BLAS threading; pin to 1 thread
# so many array tasks on the same node don't oversubscribe.
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# Must match the order in run_alpha_beta_sweep.py and build_classifications.py.
MODELS=(mixtral-8x7b mixtral-8x22b deepseek-v2-lite deepseek-v2 qwen3-30b-a3b qwen3-235b-a22b olmoe phi-3.5-moe)
DATASETS=(c4 math code wikitext2 gsm8k humaneval pile-arxiv pile-github)

ARRAY_IDX=${SLURM_ARRAY_TASK_ID:-0}
MODEL_IDX=$(( ARRAY_IDX / ${#DATASETS[@]} ))
DATASET_IDX=$(( ARRAY_IDX % ${#DATASETS[@]} ))
MODEL="${MODELS[$MODEL_IDX]}"
DATASET="${DATASETS[$DATASET_IDX]}"

echo "[$ARRAY_IDX] $MODEL/$DATASET  host=$(hostname)"

${ENV_BIN}/python experiments/build_classifications.py \
    --models "$MODEL" --datasets "$DATASET"