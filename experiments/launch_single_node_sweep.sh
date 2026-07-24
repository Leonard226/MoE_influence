#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name="sn-sweep"
#SBATCH --output=log_single_node_sweep.out

# Full single-node sweep: for each of the 6 single-node models, download
# weights -> ablate_experts.py (PPL/attn-sink/activation) -> find_super_weights.py
# -> measure_routing_shift.py -> delete weights. set -euo pipefail below means
# the FIRST failing command (download, any python script, even rm) halts the
# entire job immediately -- no later model runs after an earlier one breaks.
set -euo pipefail

ENV_BIN=/scratch/sleonard/miniconda3/envs/megatron/bin
export PATH="${ENV_BIN}:${PATH}"
export LD_LIBRARY_PATH="/scratch/sleonard/miniconda3/envs/megatron/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/scratch/sleonard/.huggingface
HUB="${HF_HOME}/hub"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/sleonard/MoE_circuits}"
cd "$PROJECT_ROOT"

# NOTE: no JSON-clearing step here (deliberately). All three scripts below
# are restart-safe -- they load the existing per-model JSON, skip any label
# already present, and only compute + append new ones. Extend DEFAULT_TARGETS
# in ablate_experts.py / find_super_weights.py to add new ablation sets;
# rerunning this script (or a single model's commands) will only compute the
# new additions, never touch or delete what's already cached.

run_model() {
    local repo_id="$1"
    local hub_dir="$2"
    local model_key="$3"
    shift 3
    local includes=("$@")

    echo "=== ${model_key}: downloading ${repo_id} ==="
    huggingface-cli download "$repo_id" --include "${includes[@]}"

    echo "=== ${model_key}: ablate_experts.py ==="
    python3 experiments/ablate_experts.py --model "$model_key"

    echo "=== ${model_key}: find_super_weights.py ==="
    python3 experiments/find_super_weights.py --model "$model_key"

    echo "=== ${model_key}: measure_routing_shift.py ==="
    python3 experiments/measure_routing_shift.py --model "$model_key"

    echo "=== ${model_key}: deleting cached weights ==="
    rm -rf "${HUB}/${hub_dir}"
}

run_model "allenai/OLMoE-1B-7B-0924" "models--allenai--OLMoE-1B-7B-0924" "olmoe" \
    "*.safetensors" "*.json" "tokenizer*"

run_model "mistralai/Mixtral-8x7B-v0.1" "models--mistralai--Mixtral-8x7B-v0.1" "mixtral-8x7b" \
    "*.safetensors" "*.json" "tokenizer*" "*.model"

run_model "Qwen/Qwen3-30B-A3B" "models--Qwen--Qwen3-30B-A3B" "qwen3-30b-a3b" \
    "*.safetensors" "*.json" "tokenizer*"

run_model "microsoft/Phi-3.5-MoE-instruct" "models--microsoft--Phi-3.5-MoE-instruct" "phi-3.5-moe" \
    "*.safetensors" "*.json" "tokenizer*"

run_model "mistralai/Mixtral-8x22B-v0.1" "models--mistralai--Mixtral-8x22B-v0.1" "mixtral-8x22b" \
    "*.safetensors" "*.json" "tokenizer*" "*.model"

run_model "deepseek-ai/DeepSeek-V2-Lite" "models--deepseek-ai--DeepSeek-V2-Lite" "deepseek-v2-lite" \
    "*.safetensors" "*.json" "tokenizer*"

echo "=== single-node sweep complete ==="
