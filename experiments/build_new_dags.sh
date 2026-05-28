#!/usr/bin/env bash
#SBATCH --nodelist=piora1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name="build-new"
#SBATCH --output=logs/build_new_dags_%j.log
# NOTE: pinned to piora1 (A100 80GB node). Other nodes (piora4, piora5)
# have V100s and will OOM on the larger models. Job queues until piora1
# frees up (currently held by the multinode build).
#
# Build DAGs for the 5 NEW datasets on 6 single-node-capable MoE models.
#
# Usage (either works):
#   sbatch experiments/build_new_dags.sh
#   # or
#   tmux new -s build_new
#   bash experiments/build_new_dags.sh 2>&1 | tee logs/build_new_dags.log
#   # detach: Ctrl-b d ;  reattach: tmux attach -t build_new
#
# IMPORTANT: each model's HuggingFace cache (~/.hugging_face/hub/models--...)
# is DELETED after all 5 datasets for that model are built. This frees disk
# space before the next model downloads -- required because the full set of
# model weights won't fit on /scratch simultaneously. Dataset caches are
# preserved (they're small and shared across models).
#
# Set CLEANUP_MODEL_CACHE=0 to disable the per-model cleanup (e.g., for a
# small-models-only test run where you want to keep weights cached).
#
# Resumable: skips (model, dataset) pairs whose output .pt already exists.
# Memory between model swaps: each `python` invocation is its own process,
# so GPU memory and resident RAM are reclaimed by the OS when it exits.
#
# qwen3-235b-a22b and deepseek-v2 need multinode SLURM; handle separately
# via experiments/launch_multinode_new.sh.

set -euo pipefail

# Find the project root by walking up from SLURM_SUBMIT_DIR (or $PWD) until we
# find config.yaml. Robust to both tmux ($0 works) and sbatch (where $0 points
# at the SLURM spool copy, not the original script).
ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$ROOT" != "/" ] && [ ! -f "$ROOT/config.yaml" ]; do
    ROOT="$(dirname "$ROOT")"
done
if [ ! -f "$ROOT/config.yaml" ]; then
    echo "ERROR: cannot find project root (no config.yaml found walking up from ${SLURM_SUBMIT_DIR:-$PWD})" >&2
    exit 1
fi
cd "$ROOT"
mkdir -p logs

# Conda env. sbatch doesn't auto-activate; prepend explicitly so `python`
# resolves to the megatron env binary in both tmux and sbatch invocations.
ENV_BIN=/scratch/sleonard/miniconda3/envs/megatron/bin
export PATH="${ENV_BIN}:${PATH}"
export LD_LIBRARY_PATH="/scratch/sleonard/miniconda3/envs/megatron/lib:${LD_LIBRARY_PATH:-}"

export HF_HOME="${HF_HOME:-$HOME/.hugging_face}"
CLEANUP_MODEL_CACHE="${CLEANUP_MODEL_CACHE:-1}"
RESULT_PATH=$(${ENV_BIN}/python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['result_path'])")

echo "ROOT=$ROOT"
echo "HF_HOME=$HF_HOME"
echo "RESULT_PATH=$RESULT_PATH"
echo "CLEANUP_MODEL_CACHE=$CLEANUP_MODEL_CACHE"
echo

NEW_DATASETS=(wikitext2 gsm8k humaneval pile-arxiv pile-github)
N_PROMPTS=1000

# Smallest first so quick wins come early.
MODELS=(
  olmoe
  phi-3.5-moe
  mixtral-8x7b
  deepseek-v2-lite
  qwen3-30b-a3b
  mixtral-8x22b
)

# Batch size per model. mixtral-8x22b is the only "large" model in this list;
# the rest are medium. BS=16 for large, BS=32 for medium.
declare -A BSZ=(
  [olmoe]=32
  [phi-3.5-moe]=32
  [mixtral-8x7b]=32
  [deepseek-v2-lite]=32
  [qwen3-30b-a3b]=32
  [mixtral-8x22b]=16
)

# HuggingFace identifier for each model (used to construct cache subdir name).
declare -A HF_ID=(
  [olmoe]="allenai/OLMoE-1B-7B-0924"
  [phi-3.5-moe]="microsoft/Phi-3.5-MoE-instruct"
  [mixtral-8x7b]="mistralai/Mixtral-8x7B-v0.1"
  [deepseek-v2-lite]="deepseek-ai/DeepSeek-V2-Lite"
  [qwen3-30b-a3b]="Qwen/Qwen3-30B-A3B"
  [mixtral-8x22b]="mistralai/Mixtral-8x22B-v0.1"
)

cleanup_model_cache() {
  # Delete $HF_HOME/hub/models--<org>--<name> for the given model.
  local m="$1"
  local id="${HF_ID[$m]}"
  local cache_path="$HF_HOME/hub/models--${id//\//--}"
  if [[ -d "$cache_path" ]]; then
    local size
    size=$(du -sh "$cache_path" 2>/dev/null | awk '{print $1}')
    echo "  cleanup: rm -rf $cache_path  (${size:-?})"
    rm -rf "$cache_path"
  else
    echo "  cleanup: no cache at $cache_path"
  fi
}

TOTAL=$(( ${#MODELS[@]} * ${#NEW_DATASETS[@]} ))
i=0
T_START=$(date +%s)

echo "Starting at $(date)"
echo "Will build $TOTAL (model, dataset) DAGs into $RESULT_PATH/circuits/"
echo "Models:   ${MODELS[*]}"
echo "Datasets: ${NEW_DATASETS[*]}"
echo "Prompts:  $N_PROMPTS each"
echo

for m in "${MODELS[@]}"; do
  for d in "${NEW_DATASETS[@]}"; do
    i=$((i + 1))
    outfile="${RESULT_PATH}/circuits/dag_${m}_${d}.pt"
    if [[ -f "$outfile" ]]; then
      echo "[$i/$TOTAL] [skip] $m/$d  (already exists)"
      continue
    fi
    elapsed_min=$(( ( $(date +%s) - T_START ) / 60 ))
    echo "============================================================"
    echo "[$i/$TOTAL] $m/$d  bs=${BSZ[$m]}  (elapsed=${elapsed_min}min)"
    echo "============================================================"
    t0=$(date +%s)
    # ABORT on any failure (download error, OOM, path-not-found, etc.)
    # to avoid the failure-chain pattern of "every model tries to download,
    # nothing works, scratch fills up". Resubmit later to retry; the
    # skip-if-exists check above resumes correctly.
    if ! ${ENV_BIN}/python experiments/build_dag.py \
        --model "$m" --dataset "$d" \
        --n_prompts $N_PROMPTS --B "${BSZ[$m]}"; then
      echo "[$i/$TOTAL] ERROR building $m/$d -- ABORTING (resubmit to retry)"
      exit 1
    fi
    dt=$(( $(date +%s) - t0 ))
    echo "[$i/$TOTAL] $m/$d done in ${dt}s"
    sleep 5  # let GPU memory fully release before next launch
  done
  echo "------------------------------------------------------------"
  echo "Finished all datasets for $m at $(date)"
  # Only clean up the model cache if EVERY expected output file exists.
  # If anything failed, leave the cache so the resubmit can reuse it
  # without re-downloading.
  all_done=1
  for d_check in "${NEW_DATASETS[@]}"; do
    if [[ ! -f "${RESULT_PATH}/circuits/dag_${m}_${d_check}.pt" ]]; then
      all_done=0
      break
    fi
  done
  if [[ "$CLEANUP_MODEL_CACHE" == "1" && "$all_done" == "1" ]]; then
    cleanup_model_cache "$m"
  elif [[ "$CLEANUP_MODEL_CACHE" == "1" ]]; then
    echo "  some $m datasets are missing; KEEPING cache (use 'rm -rf' manually if needed)"
  fi
  echo "------------------------------------------------------------"
done

T_TOTAL=$(( ( $(date +%s) - T_START ) / 60 ))
echo "============================================================"
echo "All done at $(date). Total elapsed: ${T_TOTAL} min"
echo "Reminder: qwen3-235b-a22b and deepseek-v2 need multinode SLURM. Run:"
echo "  sbatch experiments/launch_multinode_new.sh"
echo "============================================================"