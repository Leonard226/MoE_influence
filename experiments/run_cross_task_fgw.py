"""Cross-task FGW similarity for all MoE models on c4/math/code.

Resumable: saves S_matrix.npz after every completed pair, skips already-done
pairs on restart. Run in tmux for long jobs:

    tmux new -s fgw_ct
    python experiments/run_cross_task_fgw.py 2>&1 | tee /tmp/fgw_ct.log
    # detach: Ctrl-b d ; reattach: tmux attach -t fgw_ct

Outputs under {result_path}/circuits/fgw_crosstask_full/:
    S_matrix.npz   -- symmetric S matrix + key list
    heatmap.png    -- visualization
    summary.txt    -- per-model WM, per-task CMS, task-adaptiveness index, CMS-CMD
"""
import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.fgw import (
    TOKEN_CLASSES, build_token_classification, build_triple, fgw_similarity,
)
from dataset.c4_dataset   import c4_dataset_helper
from dataset.math_dataset import open_r1_math_dataset_helper
from dataset.code_dataset import code_dataset_helper

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)

SPECIAL_IDX = TOKEN_CLASSES.index("special")
MASK_THRESH = 0.5
N_PROMPTS   = 5000
MAX_TOKENS  = 32

DATASET_LOADERS = {
    "c4":   c4_dataset_helper,
    "math": open_r1_math_dataset_helper,
    "code": code_dataset_helper,
}

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

DEFAULT_MODELS = list(MODEL_IDS.keys())
DEFAULT_TASKS  = ["c4", "math", "code"]

# Cluster within-family pairs adjacently so within-family cells are easy to
# spot in the heatmap. Order: Mixtral, DeepSeek, Qwen families; then singletons.
MODEL_ORDER = [
    "mixtral-8x7b", "mixtral-8x22b",
    "deepseek-v2-lite", "deepseek-v2",
    "qwen3-30b-a3b", "qwen3-235b-a22b",
    "olmoe", "phi-3.5-moe",
]

FAMILY_PAIRS = [
    ("Mixtral 7B/22B",  "mixtral-8x7b",     "mixtral-8x22b"),
    ("DSL/DSv2",        "deepseek-v2-lite", "deepseek-v2"),
    ("Qwen 30B/235B",   "qwen3-30b-a3b",    "qwen3-235b-a22b"),
]


def mask_triple_by_class(triple, class_idx, threshold):
    """Drop vertices where class_hist[class_idx] > threshold."""
    C, F, mass, meta = triple
    keep = F[:, 4 + class_idx] <= threshold
    keep_idx = np.where(keep)[0]
    C_m = C[np.ix_(keep_idx, keep_idx)]
    F_m = F[keep_idx]
    mass_m = mass[keep_idx]; mass_m = mass_m / mass_m.sum()
    meta_m = dict(meta); meta_m["n_verts"] = len(keep_idx); meta_m["keep_idx"] = keep_idx
    return (C_m, F_m, mass_m, meta_m)


def drop_class_features(triple):
    """Slice F to [depth, out, in, load]; keep C, mass, meta unchanged."""
    C, F, mass, meta = triple
    F_short = F[:, :4]
    meta_new = dict(meta); meta_new["D"] = F_short.shape[1]
    return (C, F_short, mass, meta_new)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", default=",".join(MODEL_ORDER),
                        help="Comma-separated model names")
    parser.add_argument("--tasks", default=",".join(DEFAULT_TASKS),
                        help="Comma-separated task names")
    parser.add_argument("--alpha", type=float, default=0.0)
    parser.add_argument("--n_init", type=int, default=5)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--keep_class", action="store_true",
                        help="Keep class histogram in F (default: drop it)")
    args = parser.parse_args()

    models = args.models.split(",")
    tasks  = args.tasks.split(",")
    output_dir = args.output_dir or os.path.join(
        config["result_path"], "circuits", "fgw_crosstask_full"
    )
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    cache_dir = os.path.join(config["result_path"], "circuits", "classifications")
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    drop_class = not args.keep_class

    print(f"=== Cross-task FGW: {len(models)} models × {len(tasks)} tasks ===", flush=True)
    print(f"  models:     {models}")
    print(f"  tasks:      {tasks}")
    print(f"  output:     {output_dir}")
    print(f"  alpha={args.alpha}  n_init={args.n_init}  drop_class={drop_class}")
    print()

    # --- Verify all DAGs exist ---
    missing = []
    for m in models:
        for t in tasks:
            path = os.path.join(config["result_path"], f"circuits/dag_{m}_{t}.pt")
            if not os.path.exists(path):
                missing.append(f"{m}/{t}")
    if missing:
        print(f"ERROR: missing DAGs ({len(missing)}):")
        for x in missing:
            print(f"    {x}")
        sys.exit(1)

    # --- Load prompts (deterministic given args) ---
    print(f"=== Loading prompts ===", flush=True)
    prompts_by_task = {}
    for t in tasks:
        prompts_by_task[t] = DATASET_LOADERS[t](dataset_len=N_PROMPTS, min_words=MAX_TOKENS)
        print(f"  {t}: {len(prompts_by_task[t])} prompts")

    # --- Load DAGs ---
    print(f"\n=== Loading DAGs ===", flush=True)
    dags = {}
    for m in models:
        for t in tasks:
            path = os.path.join(config["result_path"], f"circuits/dag_{m}_{t}.pt")
            dags[(m, t)] = torch.load(path, map_location="cpu")
            print(f"  loaded {m}/{t}")

    # --- Classifications (cached) ---
    print(f"\n=== Classifications ===", flush=True)
    classifications = {}
    for m in models:
        tok = None
        for t in tasks:
            cache_path = os.path.join(cache_dir, f"classify_{m}_{t}.pkl")
            if os.path.exists(cache_path):
                with open(cache_path, "rb") as f:
                    classifications[(m, t)] = pickle.load(f)
                src = "cached"
            else:
                if tok is None:
                    tok = AutoTokenizer.from_pretrained(
                        MODEL_IDS[m], trust_remote_code=True, use_fast=True)
                print(f"  classifying {m}/{t} ...", flush=True)
                classifications[(m, t)] = build_token_classification(
                    prompts_by_task[t], tok, max_length=MAX_TOKENS, verbose=False)
                with open(cache_path, "wb") as f:
                    pickle.dump(classifications[(m, t)], f)
                src = "built"
            print(f"  {m}/{t:<5s} ({src:>6s}, {len(classifications[(m, t)]):>7,d} entries)",
                  flush=True)

    # --- Triples (mask + optional class-drop) ---
    print(f"\n=== Triples (beta=1, mask special >{MASK_THRESH}, drop_class={drop_class}) ===",
          flush=True)
    triples = {}
    for m in models:
        for t in tasks:
            tr = build_triple(dags[(m, t)], classifications[(m, t)],
                              beta=1.0, edge_threshold=0.0)
            tr = mask_triple_by_class(tr, SPECIAL_IDX, MASK_THRESH)
            if drop_class:
                tr = drop_class_features(tr)
            triples[(m, t)] = tr
            print(f"  {m}/{t:<5s}  n_verts={tr[3]['n_verts']:>5d}", flush=True)
    del dags  # free memory

    # --- Pairwise FGW (resumable) ---
    keys = [(m, t) for m in models for t in tasks]
    n_k = len(keys)
    S_path = os.path.join(output_dir, "S_matrix.npz")

    if os.path.exists(S_path):
        data = np.load(S_path, allow_pickle=True)
        S_mat = data["S_mat"]
        loaded_keys = [tuple(k) for k in data["keys"].tolist()]
        if loaded_keys != keys:
            print(f"\nWARNING: saved keys differ from current; starting fresh", flush=True)
            S_mat = np.full((n_k, n_k), np.nan)
            np.fill_diagonal(S_mat, 1.0)
        else:
            n_done = int(np.isfinite(S_mat[np.triu_indices(n_k, 1)]).sum())
            n_total = n_k * (n_k - 1) // 2
            print(f"\n=== Resumed from {S_path}: {n_done}/{n_total} pairs done ===",
                  flush=True)
    else:
        S_mat = np.full((n_k, n_k), np.nan)
        np.fill_diagonal(S_mat, 1.0)

    print(f"\n=== Pairwise FGW (alpha={args.alpha}, n_init={args.n_init}) ===", flush=True)
    n_total = n_k * (n_k - 1) // 2
    t_start = time.time()
    n_done_at_start = int(np.isfinite(S_mat[np.triu_indices(n_k, 1)]).sum())

    for i in range(n_k):
        for j in range(i + 1, n_k):
            if not np.isnan(S_mat[i, j]):
                continue
            t0 = time.time()
            try:
                S, _ = fgw_similarity(triples[keys[i]], triples[keys[j]],
                                      alpha=args.alpha, n_init=args.n_init)
                S_mat[i, j] = S
                S_mat[j, i] = S
            except Exception as e:
                print(f"  ERROR S({keys[i]} <-> {keys[j]}): {type(e).__name__}: {e}",
                      flush=True)
                S_mat[i, j] = -1.0  # sentinel for failure
                S_mat[j, i] = -1.0

            n_done = int(np.isfinite(S_mat[np.triu_indices(n_k, 1)]).sum())
            elapsed = time.time() - t_start
            pairs_done = n_done - n_done_at_start
            if pairs_done > 0:
                rate = pairs_done / elapsed
                remaining_min = (n_total - n_done) / rate / 60 if rate > 0 else 0
                eta_str = f"ETA={remaining_min:.1f}min"
            else:
                eta_str = "ETA=?"
            dt = time.time() - t0
            print(f"  [{n_done:>3d}/{n_total}] "
                  f"S({keys[i][0]:<20s}/{keys[i][1]:<5s} <-> "
                  f"{keys[j][0]:<20s}/{keys[j][1]:<5s}) = {S_mat[i, j]:.4f}  "
                  f"[{dt:.1f}s, {eta_str}]", flush=True)

            # Checkpoint after every pair.
            np.savez(S_path, S_mat=S_mat,
                     keys=np.array(keys, dtype=object))

    print(f"\n=== All pairs done in {(time.time()-t_start)/60:.1f} min ===", flush=True)

    # --- Heatmap ---
    labels = [f"{m}/{t}" for m, t in keys]
    fig, ax = plt.subplots(figsize=(14, 13))
    S_show = np.where(S_mat < 0, np.nan, S_mat)
    im = ax.imshow(S_show, cmap="viridis", vmin=0.2, vmax=1.0, origin="upper")
    ax.set_xticks(range(n_k)); ax.set_yticks(range(n_k))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    for i in range(n_k):
        for j in range(n_k):
            if not np.isfinite(S_show[i, j]):
                ax.text(j, i, "X", ha="center", va="center", color="red", fontsize=6)
            else:
                ax.text(j, i, f"{S_show[i, j]:.2f}", ha="center", va="center",
                        color="white" if S_show[i, j] < 0.6 else "black", fontsize=6)
    # Within-model blocks (red).
    for mi in range(len(models)):
        base = mi * len(tasks)
        ax.add_patch(plt.Rectangle((base - 0.5, base - 0.5), len(tasks), len(tasks),
                                   fill=False, edgecolor="red", lw=1.5))
    # Cross-model same-task cells (blue).
    for ti in range(len(tasks)):
        for mi in range(len(models)):
            for mj in range(mi + 1, len(models)):
                ri, rj = mi * len(tasks) + ti, mj * len(tasks) + ti
                ax.add_patch(plt.Rectangle((rj - 0.5, ri - 0.5), 1, 1,
                                           fill=False, edgecolor="blue", lw=0.8))
                ax.add_patch(plt.Rectangle((ri - 0.5, rj - 0.5), 1, 1,
                                           fill=False, edgecolor="blue", lw=0.8))
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04,
                 label=f"S(alpha={args.alpha})")
    ax.set_title(f"Cross-task FGW: {len(models)} models × {len(tasks)} tasks  "
                 f"(alpha={args.alpha}, special-masked, "
                 f"{'no class' if drop_class else 'with class'})")
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "heatmap.png")
    plt.savefig(heatmap_path, dpi=120)
    plt.close(fig)
    print(f"Heatmap saved: {heatmap_path}")

    # --- Block summaries (text) ---
    lines = []
    lines.append(f"=== Cross-task FGW summary ===")
    lines.append(f"  {len(models)} models × {len(tasks)} tasks, alpha={args.alpha}, "
                 f"n_init={args.n_init}, drop_class={drop_class}")
    lines.append("")

    # Per-model WM
    lines.append("=== Per-model WM (within-model cross-task mean) ===")
    wm_per_model = {}
    for mi, m in enumerate(models):
        base = mi * len(tasks)
        pairs = []
        per_pair_text = []
        for ti in range(len(tasks)):
            for tj in range(ti + 1, len(tasks)):
                v = S_mat[base + ti, base + tj]
                if np.isfinite(v) and v >= 0:
                    pairs.append(v)
                per_pair_text.append(f"{tasks[ti]}↔{tasks[tj]}={v:.3f}")
        wm_per_model[m] = float(np.mean(pairs)) if pairs else float("nan")
        lines.append(f"  {m:<22s}  WM={wm_per_model[m]:.4f}  [{'  '.join(per_pair_text)}]")

    # Per-task CMS
    lines.append("\n=== Per-task CMS (cross-model same-task mean) ===")
    cms_per_task = {}
    for ti, t in enumerate(tasks):
        same_t = []
        for mi in range(len(models)):
            for mj in range(mi + 1, len(models)):
                v = S_mat[mi*len(tasks) + ti, mj*len(tasks) + ti]
                if np.isfinite(v) and v >= 0:
                    same_t.append(v)
        cms_per_task[t] = float(np.mean(same_t)) if same_t else float("nan")
        lines.append(f"  {t:<6s}  CMS={cms_per_task[t]:.4f}  "
                     f"min={min(same_t):.4f}  max={max(same_t):.4f}")

    # Task-adaptiveness index
    lines.append("\n=== Per-model task-adaptiveness  (WM - mean CMS-with-others) ===")
    lines.append(f"  {'Model':<22s}  {'WM':>7s}  {'CMS-others':>11s}  {'WM-CMS':>8s}  label")
    for mi, m in enumerate(models):
        base = mi * len(tasks)
        wm = wm_per_model[m]
        cms_others = []
        for mj in range(len(models)):
            if mj == mi: continue
            for ti in range(len(tasks)):
                v = S_mat[base + ti, mj*len(tasks) + ti]
                if np.isfinite(v) and v >= 0:
                    cms_others.append(v)
        cms_o = float(np.mean(cms_others)) if cms_others else float("nan")
        diff = wm - cms_o
        if diff > 0.05:
            label = "task-invariant"
        elif diff < -0.05:
            label = "task-adaptive"
        else:
            label = "borderline"
        lines.append(f"  {m:<22s}  {wm:>7.4f}  {cms_o:>11.4f}  {diff:>+8.4f}  [{label}]")

    # Global CMS vs CMD
    cms_all, cmd_all = [], []
    for mi in range(len(models)):
        for mj in range(mi + 1, len(models)):
            for ti in range(len(tasks)):
                for tj in range(len(tasks)):
                    v = S_mat[mi*len(tasks) + ti, mj*len(tasks) + tj]
                    if np.isfinite(v) and v >= 0:
                        (cms_all if ti == tj else cmd_all).append(v)
    if cms_all and cmd_all:
        cms_mean = float(np.mean(cms_all))
        cmd_mean = float(np.mean(cmd_all))
        lines.append("\n=== Global CMS vs CMD ===")
        lines.append(f"  CMS mean (n={len(cms_all):>3d}): {cms_mean:.4f}")
        lines.append(f"  CMD mean (n={len(cmd_all):>3d}): {cmd_mean:.4f}")
        lines.append(f"  CMS - CMD = {cms_mean - cmd_mean:+.4f}")

    # Within-family
    lines.append("\n=== Within-family same-task ===")
    for name, ma, mb in FAMILY_PAIRS:
        if ma in models and mb in models:
            ia = models.index(ma); ib = models.index(mb)
            for ti, t in enumerate(tasks):
                v = S_mat[ia*len(tasks) + ti, ib*len(tasks) + ti]
                if np.isfinite(v) and v >= 0:
                    lines.append(f"  {name:<18s}  S({ma}/{t}, {mb}/{t}) = {v:.4f}")

    summary_text = "\n".join(lines)
    print("\n" + summary_text)
    summary_path = os.path.join(output_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write(summary_text + "\n")
    print(f"\nSummary saved: {summary_path}")


if __name__ == "__main__":
    main()
