"""Build the spaCy-POS-based token classification cache for every (model, dataset).

For every combination of MODEL and DATASET, ensures
    ${result_path}/circuits/classifications/classify_<model>_<dataset>.pkl
exists. Skips pairs that are already built. Loads the HF tokenizer per model
(small download per first use), re-creates the exact prompt set that
build_dag.py used (deterministic given dataset_len + min_words), and runs
build_token_classification.

CPU-only. No GPU required. Run on any CPU node:

    python experiments/build_classifications.py

Or sbatch it on a low-priority CPU partition (no GPU). HumanEval has only
164 prompts; for it we cap dataset_len to whatever the loader returned (the
batched DAG build already capped N_PROMPTS in the same way).
"""
import argparse
import importlib
import os
import pickle
import sys
import time
from pathlib import Path

import yaml
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.fgw import build_token_classification, TOKEN_CLASSES

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)

# Same model + dataset registries used by build_dag.py and the sweep.
MODELS = [
    "mixtral-8x7b", "mixtral-8x22b",
    "deepseek-v2-lite", "deepseek-v2",
    "qwen3-30b-a3b", "qwen3-235b-a22b",
    "olmoe", "phi-3.5-moe",
]
DATASETS = [
    "c4", "math", "code",
    "wikitext2", "gsm8k", "humaneval",
    "pile-arxiv", "pile-github",
]

MODEL_IDS = {
    "olmoe":            "allenai/OLMoE-1B-7B-0924",
    "deepseek-v2":      "deepseek-ai/DeepSeek-V2",
    "deepseek-v2-lite": "deepseek-ai/DeepSeek-V2-Lite",
    "mixtral-8x7b":     "mistralai/Mixtral-8x7B-v0.1",
    "mixtral-8x22b":    "mistralai/Mixtral-8x22B-v0.1",
    "qwen3-30b-a3b":    "Qwen/Qwen3-30B-A3B",
    "qwen3-235b-a22b":  "Qwen/Qwen3-235B-A22B",
    "phi-3.5-moe":      "microsoft/Phi-3.5-MoE-instruct",
}

# Match build_dag.py / build_dag_multinode.py.
DATASET_LOADERS = {
    "c4":          ("dataset.c4_dataset",          "c4_dataset_helper"),
    "math":        ("dataset.math_dataset",        "open_r1_math_dataset_helper"),
    "code":        ("dataset.code_dataset",        "code_dataset_helper"),
    "wikitext2":   ("dataset.wikitext2_dataset",   "wikitext2_dataset_helper"),
    "gsm8k":       ("dataset.gsm8k_dataset",       "gsm8k_dataset_helper"),
    "humaneval":   ("dataset.humaneval_dataset",   "humaneval_dataset_helper"),
    "pile-arxiv":  ("dataset.pile_arxiv_dataset",  "pile_arxiv_dataset_helper"),
    "pile-github": ("dataset.pile_github_dataset", "pile_github_dataset_helper"),
}

# Prompt count + truncation: must match what was used at DAG-build time so the
# classification's (prompt_idx, position) keys line up with top_prompt/top_pos
# entries in the DAG.
# c4/math/code: built with 5000 prompts.
# new datasets: built with 1000 prompts; humaneval capped to actual length.
N_PROMPTS = {
    "c4":          5000,
    "math":        5000,
    "code":        5000,
    "wikitext2":   1000,
    "gsm8k":       1000,
    "humaneval":   1000,  # loader caps internally at 164
    "pile-arxiv":  1000,
    "pile-github": 1000,
}
MAX_TOKENS = 32

CACHE_DIR = os.path.join(config["result_path"], "circuits", "classifications")
Path(CACHE_DIR).mkdir(parents=True, exist_ok=True)


def load_prompts(dataset):
    mod_name, fn_name = DATASET_LOADERS[dataset]
    loader = getattr(importlib.import_module(mod_name), fn_name)
    return loader(dataset_len=N_PROMPTS[dataset], min_words=MAX_TOKENS)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODELS),
                        help="Comma-separated subset of models (default: all)")
    parser.add_argument("--datasets", default=",".join(DATASETS),
                        help="Comma-separated subset of datasets (default: all)")
    args = parser.parse_args()

    models = args.models.split(",")
    datasets = args.datasets.split(",")

    # Cache prompts per dataset so we only call the loader once even if 8
    # models all need it.
    prompts_cache = {}

    todo = []
    for m in models:
        for d in datasets:
            cache_path = os.path.join(CACHE_DIR, f"classify_{m}_{d}.pkl")
            if os.path.exists(cache_path):
                continue
            todo.append((m, d, cache_path))

    print(f"=== build_classifications.py ===")
    print(f"  cache dir : {CACHE_DIR}")
    print(f"  models    : {len(models)}  datasets: {len(datasets)}")
    print(f"  already   : {len(models) * len(datasets) - len(todo)}")
    print(f"  to build  : {len(todo)}")
    print()
    if not todo:
        print("All classifications already built. Nothing to do.")
        return

    t_start = time.time()
    for i, (m, d, cache_path) in enumerate(todo, 1):
        elapsed_min = (time.time() - t_start) / 60
        print(f"[{i}/{len(todo)}] {m}/{d}  (elapsed={elapsed_min:.1f}min)", flush=True)
        t0 = time.time()

        # Load prompts (cached per-dataset).
        if d not in prompts_cache:
            print(f"  loading {d} prompts ({N_PROMPTS[d]} requested) ...", flush=True)
            prompts_cache[d] = load_prompts(d)
        prompts = prompts_cache[d]
        print(f"  using {len(prompts)} prompts")

        # Tokenizer.
        print(f"  loading tokenizer {MODEL_IDS[m]} ...", flush=True)
        tok = AutoTokenizer.from_pretrained(
            MODEL_IDS[m], trust_remote_code=True, use_fast=True,
        )

        # Classify.
        print(f"  classifying ...", flush=True)
        classification = build_token_classification(
            prompts, tok, max_length=MAX_TOKENS, verbose=False,
        )

        # Save.
        with open(cache_path, "wb") as f:
            pickle.dump(classification, f)
        print(f"  saved {cache_path}  ({len(classification):,} entries, {time.time() - t0:.1f}s)",
              flush=True)

    print(f"\nAll done in {(time.time() - t_start) / 60:.1f} min.")


if __name__ == "__main__":
    main()