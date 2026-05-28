#!/bin/bash
#SBATCH --nodelist=piora1,piora2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name="MN-newdsets"
#SBATCH --output=logs/build_multinode_new.log
#
# Build DAGs for the multinode-only models (qwen3-235b-a22b, deepseek-v2)
# on the 5 NEW datasets. Submit:
#   sbatch experiments/launch_multinode_new.sh
#
# Resumable: skips (model, dataset) pairs whose output .pt already exists.
# All 2 models × 5 datasets = 10 builds run sequentially in this allocation;
# a failure on one pair doesn't block the next.

set -euo pipefail

ENV_BIN=/scratch/sleonard/miniconda3/envs/megatron/bin
export PATH="${ENV_BIN}:${PATH}"
export LD_LIBRARY_PATH="/scratch/sleonard/miniconda3/envs/megatron/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME="${HF_HOME:-$HOME/.hugging_face}"

# Resolve master node from SLURM env.
nodes=( $(scontrol show hostnames "$SLURM_JOB_NODELIST") )
export MASTER_ADDR="${nodes[0]}"
export MASTER_PORT=29500
echo "Python: $(${ENV_BIN}/python --version)  torchrun: ${ENV_BIN}/torchrun"
echo "MASTER_ADDR=$MASTER_ADDR  MASTER_PORT=$MASTER_PORT  nodes=${nodes[*]}  HF_HOME=$HF_HOME"

export NCCL_IB_DISABLE=0
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/sleonard/MoE_circuits}"

# Locate the multinode build script (handle either layout).
if [ -f "${PROJECT_ROOT}/experiments/build_dag_multinode.py" ]; then
    SCRIPT_PATH="${PROJECT_ROOT}/experiments/build_dag_multinode.py"
elif [ -f "${PROJECT_ROOT}/experiments/circuits/build_dag_multinode.py" ]; then
    SCRIPT_PATH="${PROJECT_ROOT}/experiments/circuits/build_dag_multinode.py"
else
    echo "ERROR: build_dag_multinode.py not found under ${PROJECT_ROOT}/experiments/" >&2
    exit 1
fi
echo "SCRIPT_PATH=$SCRIPT_PATH"

RESULT_PATH=$(${ENV_BIN}/python -c "import yaml; print(yaml.safe_load(open('${PROJECT_ROOT}/config.yaml'))['result_path'])")

# Outer loop: models that need multinode. Inner loop: 5 new datasets. BS=16.
MODELS=(qwen3-235b-a22b deepseek-v2)
DATASETS=(wikitext2 gsm8k humaneval pile-arxiv pile-github)

for MODEL in "${MODELS[@]}"; do
    for DATASET in "${DATASETS[@]}"; do
        OUTFILE="${RESULT_PATH}/circuits/dag_${MODEL}_${DATASET}.pt"
        if [ -f "$OUTFILE" ]; then
            echo "[skip] ${MODEL}/${DATASET} already exists"
            continue
        fi
        echo "================================================================"
        echo "Starting ${MODEL}/${DATASET} at $(date)"
        echo "================================================================"
        srun --export=ALL ${ENV_BIN}/torchrun \
            --nnodes=2 \
            --nproc_per_node=4 \
            --node_rank=$SLURM_NODEID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            "$SCRIPT_PATH" \
            --model "$MODEL" \
            --dataset "$DATASET" \
            --n_prompts 1000 \
            --B 16 \
            || echo "WARN: ${MODEL}/${DATASET} failed; continuing with next"
        echo "Finished ${MODEL}/${DATASET} at $(date)"
        # Let port 29500 clear TIME_WAIT before next torchrun rendezvous.
        sleep 30
    done
done

echo "All done at $(date)"