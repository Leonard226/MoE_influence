#!/bin/bash
#SBATCH --nodelist=piora1,piora2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name="build-dags-mn"
#SBATCH --output=logs/build_dag_multinode_%j.log
#
# Build DAGs for the multinode-only models (qwen3-235b-a22b, deepseek-v2).
# Counterpart to build_dags.sh, which covers the 6 single-node models.
#
# Usage:
#   Full sweep -- both models x all 8 datasets:
#     sbatch experiments/build_dag_multinode.sh
#   One model, all datasets:
#     sbatch experiments/build_dag_multinode.sh --model deepseek-v2
#   One (model, dataset) pair, in isolation:
#     sbatch experiments/build_dag_multinode.sh --model deepseek-v2 --dataset c4
#
# This needs an actual multi-node SLURM allocation (srun/torchrun,
# $SLURM_JOB_NODELIST, $SLURM_NODEID) -- unlike build_dags.sh it cannot be
# run directly via tmux/bash, only via sbatch.
#
# IMPORTANT: in full-sweep mode, each model's HuggingFace cache
# ($HF_HOME/hub/models--...) is DELETED after all its datasets are built,
# to free disk before the next model downloads -- the full set of model
# weights won't fit on /scratch simultaneously. Dataset caches are
# preserved. This cleanup defaults OFF when --model restricts to a single
# model (you're likely iterating on it and don't want to force a
# re-download) -- pass CLEANUP_MODEL_CACHE=1 to force it on, or =0 to force
# it off in full-sweep mode.
#
# Resumable: skips (model, dataset) pairs whose output .pt already exists.

set -euo pipefail

ALL_MODELS=(deepseek-v2 qwen3-235b-a22b)
ALL_DATASETS=(c4 math code wikitext2 gsm8k humaneval pile-arxiv pile-github)

usage() {
  cat <<EOF
Usage: $0 [--model MODEL] [--dataset DATASET]

  --model MODEL      Restrict to this model (default: all).
                      One of: ${ALL_MODELS[*]}
  --dataset DATASET   Restrict to this dataset (default: all).
                      One of: ${ALL_DATASETS[*]}
  -h, --help          Show this help and exit.

Examples:
  $0                                       # both models x all datasets
  $0 --model deepseek-v2                   # one model, all datasets
  $0 --model deepseek-v2 --dataset c4      # one (model, dataset) pair
EOF
}

ONLY_MODEL=""
ONLY_DATASET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)   ONLY_MODEL="${2:-}"; shift 2 ;;
    --dataset) ONLY_DATASET="${2:-}"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "ERROR: unknown argument '$1'" >&2; usage; exit 1 ;;
  esac
done

if [[ -n "$ONLY_MODEL" ]]; then
  found=0
  for m in "${ALL_MODELS[@]}"; do [[ "$m" == "$ONLY_MODEL" ]] && found=1; done
  if [[ "$found" == "0" ]]; then
    echo "ERROR: --model '$ONLY_MODEL' is not one of: ${ALL_MODELS[*]}" >&2
    exit 1
  fi
fi
if [[ -n "$ONLY_DATASET" ]]; then
  found=0
  for d in "${ALL_DATASETS[@]}"; do [[ "$d" == "$ONLY_DATASET" ]] && found=1; done
  if [[ "$found" == "0" ]]; then
    echo "ERROR: --dataset '$ONLY_DATASET' is not one of: ${ALL_DATASETS[*]}" >&2
    exit 1
  fi
fi

ENV_BIN=/scratch/sleonard/miniconda3/envs/megatron/bin
export PATH="${ENV_BIN}:${PATH}"
export LD_LIBRARY_PATH="/scratch/sleonard/miniconda3/envs/megatron/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.hugging_face}"

# Default: cache cleanup on for a full sweep, off when restricted to one model
# (see header comment). CLEANUP_MODEL_CACHE env var always overrides.
CLEANUP_MODEL_CACHE="${CLEANUP_MODEL_CACHE:-$([[ -n "$ONLY_MODEL" ]] && echo 0 || echo 1)}"

# Resolve master node from SLURM env.
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT=29500
echo "Python: $(${ENV_BIN}/python --version)  torchrun: ${ENV_BIN}/torchrun"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  nodes=${nodes[*]}"
echo "HF_HOME=$HF_HOME  CLEANUP_MODEL_CACHE=$CLEANUP_MODEL_CACHE"

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# Find the project root: walk up from SLURM_SUBMIT_DIR until we hit a
# directory containing config.yaml. This is robust to whether you submit
# from the repo root or from experiments/.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$PWD}"
while [ "$PROJECT_ROOT" != "/" ] && [ ! -f "$PROJECT_ROOT/config.yaml" ]; do
    PROJECT_ROOT="$(dirname "$PROJECT_ROOT")"
done
if [ ! -f "$PROJECT_ROOT/config.yaml" ]; then
    echo "ERROR: cannot find project root (no config.yaml found walking up from ${SLURM_SUBMIT_DIR:-$PWD})" >&2
    exit 1
fi
echo "PROJECT_ROOT=$PROJECT_ROOT"

SCRIPT_PATH="${PROJECT_ROOT}/experiments/build_dag_multinode.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: $SCRIPT_PATH not found" >&2
    exit 1
fi
echo "SCRIPT_PATH=$SCRIPT_PATH"

RESULT_PATH=$(${ENV_BIN}/python -c "import yaml; print(yaml.safe_load(open('${PROJECT_ROOT}/config.yaml'))['result_path'])")

# HF identifiers (for cache cleanup). ALL_MODELS is ordered deepseek-v2 first
# because its weights are already in cache; running qwen first would mean
# downloading qwen while deepseek is still on disk (peak = both). After
# deepseek finishes and its cache is deleted, qwen downloads onto the freed
# space. MODELS/DATASETS: the full lists by default, or a single entry when
# restricted via --model/--dataset.
declare -A HF_ID=(
  [deepseek-v2]="deepseek-ai/DeepSeek-V2"
  [qwen3-235b-a22b]="Qwen/Qwen3-235B-A22B"
)
if [[ -n "$ONLY_MODEL" ]]; then
  MODELS=("$ONLY_MODEL")
else
  MODELS=("${ALL_MODELS[@]}")
fi
if [[ -n "$ONLY_DATASET" ]]; then
  DATASETS=("$ONLY_DATASET")
else
  DATASETS=("${ALL_DATASETS[@]}")
fi

cleanup_model_cache() {
  local m="$1"
  local id="${HF_ID[$m]}"
  local cache_path="$HF_HOME/hub/models--${id//\//--}"
  if [ -d "$cache_path" ]; then
    local size
    size=$(du -sh "$cache_path" 2>/dev/null | awk '{print $1}')
    echo "  cleanup: rm -rf $cache_path  (${size:-?})"
    rm -rf "$cache_path"
  else
    echo "  cleanup: no cache at $cache_path"
  fi
}

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        OUTFILE="${RESULT_PATH}/dags/${DATASET}/dag_${MODEL}_${DATASET}.pt"
        if [ -f "$OUTFILE" ]; then
            echo "[skip] ${MODEL}/${DATASET} already exists"
            continue
        fi
        echo "================================================================"
        echo "Starting ${MODEL}/${DATASET} at $(date)"
        echo "================================================================"
        # ABORT on any failure (download error, OOM, path-not-found, NCCL,
        # missing package, etc.) to avoid the failure-chain pattern of
        # "every model tries to download, nothing works, scratch fills up".
        # Resubmit later to retry; the skip-if-exists check above resumes
        # correctly.
        if ! srun --export=ALL ${ENV_BIN}/torchrun \
            --nnodes=2 \
            --nproc_per_node=4 \
            --node_rank=$SLURM_NODEID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            "$SCRIPT_PATH" \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --n_prompts 500 \
            --B 16
        then
            echo "ERROR: ${MODEL}/${DATASET} failed -- ABORTING (resubmit to retry)"
            exit 1
        fi
        echo "Finished ${MODEL}/${DATASET} at $(date)"
        sleep 30  # let port 29500 clear TIME_WAIT
    done
    echo "----------------------------------------------------------------"
    echo "Finished all datasets for ${MODEL} at $(date)"
    # Only clean up the model cache if EVERY expected output file exists.
    # If anything failed, leave the cache so the resubmit can reuse it
    # without re-downloading.
    all_done=1
    for D_CHECK in "${DATASETS[@]}"; do
        if [ ! -f "${RESULT_PATH}/dags/${D_CHECK}/dag_${MODEL}_${D_CHECK}.pt" ]; then
            all_done=0
            break
        fi
    done
    if [ "$CLEANUP_MODEL_CACHE" = "1" ] && [ "$all_done" = "1" ]; then
        cleanup_model_cache "$MODEL"
    elif [ "$CLEANUP_MODEL_CACHE" = "1" ]; then
        echo "  some ${MODEL} datasets are missing; KEEPING cache (use 'rm -rf' manually if needed)"
    fi
    echo "----------------------------------------------------------------"
done

echo "All done at $(date)"