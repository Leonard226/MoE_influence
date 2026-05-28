#!/usr/bin/env bash
#
# Build DAGs for the 5 NEW datasets on 7 single-node-capable MoE models.
#
# Usage:
#   tmux new -s build_new
#   bash experiments/build_new_dags.sh 2>&1 | tee logs/build_new_dags.log
#   # detach: Ctrl-b d ;  reattach: tmux attach -t build_new
#
# Resumable: skips (model, dataset) pairs whose output .pt already exists.
# Memory freed between models: each `python` invocation is its own process,
# so GPU memory and resident RAM are reclaimed by the OS when it exits. A
# 5-second sleep is added between runs to let device buffers fully release.
#
# DeepSeek-V2 (236B) needs multi-node SLURM allocation; handle separately
# via experiments/launch_multinode_new.sh.

set -u

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
mkdir -p logs

RESULT_PATH=$(python -c "import yaml; print(yaml.safe_load(open('config.yaml'))['result_path'])")

NEW_DATASETS=(wikitext2 gsm8k humaneval pile-arxiv pile-github)
N_PROMPTS=1000

# Smallest first so quick wins come early. qwen3-235b-a22b and deepseek-v2
# are omitted -- both need multinode and are handled by launch_multinode_new.sh.
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
    if ! python experiments/build_dag.py \
        --model "$m" --dataset "$d" \
        --n_prompts $N_PROMPTS --B "${BSZ[$m]}"; then
      echo "[$i/$TOTAL] ERROR building $m/$d (continuing with next)"
    fi
    dt=$(( $(date +%s) - t0 ))
    echo "[$i/$TOTAL] $m/$d done in ${dt}s"
    sleep 5  # let GPU memory fully release before next launch
  done
done

T_TOTAL=$(( ( $(date +%s) - T_START ) / 60 ))
echo "============================================================"
echo "All done at $(date). Total elapsed: ${T_TOTAL} min"
echo "Reminder: qwen3-235b-a22b and deepseek-v2 need multinode SLURM. Run:"
echo "  sbatch experiments/launch_multinode_new.sh"
echo "============================================================"
