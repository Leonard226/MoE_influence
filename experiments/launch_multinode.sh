#!/bin/bash
#SBATCH --nodelist=piora1,piora2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name="CR7"
#SBATCH --output=log.out

set -euo pipefail
MODEL="${MODEL:-deepseek-v2}"
# Use the megatron env's binaries directly. torchrun's shebang already points at
# the correct python interpreter inside the env, so no conda activation needed.
ENV_BIN=/scratch/sleonard/miniconda3/envs/megatron/bin
export PATH="${ENV_BIN}:${PATH}"
# Some bnb / nccl libs live in the env's lib dir; make them findable.
export LD_LIBRARY_PATH="/scratch/sleonard/miniconda3/envs/megatron/lib:${LD_LIBRARY_PATH:-}"

# Preserve the HuggingFace cache location (set in user's login shell, lost in srun).
export HF_HOME="${HF_HOME:-$HOME/.hugging_face}"

# Resolve master node IP from SLURM env.
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT=29500
echo "Python: $(${ENV_BIN}/python --version)  torchrun: ${ENV_BIN}/torchrun"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  nodes=${nodes[*]}  HF_HOME=$HF_HOME"

# NCCL — favor IB; suppress noisy debug unless explicitly enabled.
export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

# sbatch copies the .sh to /cm/local/apps/slurm/var/spool/job{ID}/ before executing,
# so BASH_SOURCE[0] points there — not at the original launch_multinode.sh location.
# Use SLURM_SUBMIT_DIR (the dir where `sbatch` was invoked) as the project root.
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/sleonard/MoE_circuits}"
SCRIPT_PATH="${PROJECT_ROOT}/experiments/circuits/build_dag_multinode.py"

if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: build_dag_multinode.py not found at $SCRIPT_PATH" >&2
    echo "       SLURM_SUBMIT_DIR=$SLURM_SUBMIT_DIR" >&2
    echo "       Run: sbatch from /scratch/sleonard/MoE_circuits/" >&2
    exit 1
fi
echo "SCRIPT_PATH=$SCRIPT_PATH"

# Note: --ntasks-per-node=1 means srun spawns one task per node; that task itself
# runs torchrun, which fans out 4 worker processes (one per GPU on the node).
# Run c4, math, code sequentially in one allocation. Datasets are independent,
# so a failure in one doesn't block the next.
for DATASET in c4 math code; do
    echo "==================================================="
    echo "Starting dataset=${DATASET} at $(date)"
    echo "==================================================="
    srun --export=ALL ${ENV_BIN}/torchrun \
        --nnodes=2 \
        --nproc_per_node=4 \
        --node_rank=$SLURM_NODEID \
        --rdzv_backend=c10d \
        --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
        "$SCRIPT_PATH" \
        --model $MODEL \
        --dataset "$DATASET" \
        --n_prompts 1000 \
        --B 16 \
        || echo "WARN: dataset=${DATASET} failed; continuing"
    echo "Finished dataset=${DATASET} at $(date)"
    # Brief pause so port 29500 (c10d rendezvous) clears TIME_WAIT before re-bind.
    sleep 30
done
