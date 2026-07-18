#!/bin/bash
#SBATCH --nodelist=piora1,piora2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name="ablate-mn"
#SBATCH --output=log_ablate_mn.out

# Multi-node expert ablation for qwen3-235b-a22b / deepseek-v2.
# Usage: MODEL=qwen3-235b-a22b sbatch experiments/launch_ablate_multinode.sh
# Append custom ablations to the existing JSON (restart-safe; cached keys are
# skipped, no random controls added):
#   MODEL=deepseek-v2 EXPERTS="LaEb+LcEd;LeEf" sbatch experiments/launch_ablate_multinode.sh
set -euo pipefail
MODEL="${MODEL:-qwen3-235b-a22b}"
EXPERTS="${EXPERTS:-}"
# PPL batch size. Eager attention is O(seq^2 * heads) per layer at seq=2048;
# DeepSeek-V2 (~128 heads, hidden 5120) OOMs at 4, so default it to 1.
# Batch size does NOT change PPL (per-sequence NLL), only memory/speed.
if [ "$MODEL" = "deepseek-v2" ]; then BATCH="${BATCH:-1}"; else BATCH="${BATCH:-4}"; fi

ENV_BIN=/scratch/sleonard/miniconda3/envs/megatron/bin
export PATH="${ENV_BIN}:${PATH}"
export LD_LIBRARY_PATH="/scratch/sleonard/miniconda3/envs/megatron/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.hugging_face}"

nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR="${nodes[0]}"
# Derive a unique port per job so a stale TIME_WAIT socket from a previous run
# can never collide with this one's c10d rendezvous (fixed 29500 caused a
# RendezvousConnectionError when reused right after another multinode job).
export MASTER_PORT=$(( 20000 + SLURM_JOB_ID % 10000 ))
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  nodes=${nodes[*]}  MODEL=$MODEL"

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/sleonard/MoE_circuits}"
SCRIPT_PATH="${PROJECT_ROOT}/experiments/ablate_experts_multinode.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: ablate_experts_multinode.py not found at $SCRIPT_PATH" >&2
    exit 1
fi

# When EXPERTS is set, append exactly those ablations (no random controls);
# otherwise run the model's built-in default target list.
EXTRA_ARGS=()
if [ -n "$EXPERTS" ]; then
    EXTRA_ARGS+=(--experts "$EXPERTS" --random-controls 0)
fi

srun --export=ALL ${ENV_BIN}/torchrun \
    --nnodes=2 \
    --nproc_per_node=4 \
    --node_rank=$SLURM_NODEID \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    "$SCRIPT_PATH" \
    --model "$MODEL" \
    --batch-size "$BATCH" \
    "${EXTRA_ARGS[@]}"
