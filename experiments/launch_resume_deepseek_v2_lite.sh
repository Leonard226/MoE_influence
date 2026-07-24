#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=4
#SBATCH --job-name="resume-dsv2l"
#SBATCH --output=log_resume_deepseek_v2_lite.out
#SBATCH --time=04:00:00

# One-off resume: launch_single_node_sweep.sh crashed inside
# measure_routing_shift.py for deepseek-v2-lite (gate hook bug, now fixed
# in measure_routing_shift.py). ablate_experts.py and find_super_weights.py
# already completed successfully for this model in that run -- do NOT
# re-run the full sweep script, its JSON-clearing loop would wipe the 5
# other models' already-completed results and force needless re-downloads.
set -euo pipefail

ENV_BIN=/scratch/sleonard/miniconda3/envs/megatron/bin
export PATH="${ENV_BIN}:${PATH}"
export LD_LIBRARY_PATH="/scratch/sleonard/miniconda3/envs/megatron/lib:${LD_LIBRARY_PATH:-}"
export HF_HOME=/scratch/sleonard/.huggingface
HUB="${HF_HOME}/hub"

PROJECT_ROOT="${SLURM_SUBMIT_DIR:-/scratch/sleonard/MoE_circuits}"
cd "$PROJECT_ROOT"

echo "=== deepseek-v2-lite: measure_routing_shift.py ==="
python3 experiments/measure_routing_shift.py --model deepseek-v2-lite

echo "=== deepseek-v2-lite: deleting cached weights ==="
rm -rf "${HUB}/models--deepseek-ai--DeepSeek-V2-Lite"

echo "=== resume complete ==="
