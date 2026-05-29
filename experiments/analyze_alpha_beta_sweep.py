"""Headline analysis of the alpha-beta sweep results.

Reads S_full.npz produced by aggregate_alpha_beta_sweep.py and writes
several headline outputs to {result_path}/circuits/alpha_beta_sweep/analysis/:

    summary.txt
        Coverage, global aggregates, per-task CMS, per-model WM, within-family
        pair tables, α-axis decomposition.
    heatmap_alpha_grid.png
        5 x 5 grid of mini-heatmaps -- the (α, β) landscape at a glance.
    heatmap_pure_feature.png    (α=0)
    heatmap_balanced.png        (α=0.5, β=0.5)
    heatmap_pure_structure.png  (α=1, β=0.5)
        Full 64 x 64 heatmaps with model-block gridlines and within-family
        same-task cells highlighted (red boxes).
    within_family_pairs.png
        For each family (Mixtral 7B/22B, DSL/DSv2, Qwen 30B/235B), the bar
        chart of within-family same-task S across all 8 tasks at α=0, β=0.5.
    cms_per_task.png
        Per-task CMS (mean cross-model same-task similarity) at α=0, β=0.5.
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import yaml
from pathlib import Path

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.run_alpha_beta_sweep import (  # noqa: E402
    MODELS, TASKS, TUPLES, N_TUPLES, ALPHAS, BETAS,
)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)

INPUT_DIR = os.path.join(config["result_path"], "circuits", "alpha_beta_sweep")
INPUT_PATH = os.path.join(INPUT_DIR, "S_full.npz")
OUT_DIR = os.path.join(INPUT_DIR, "analysis")
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Family structure (for within/cross-family decomposition).
# ---------------------------------------------------------------------------
FAMILIES = {
    "Mixtral": ["mixtral-8x7b",     "mixtral-8x22b"],
    "DeepSeek": ["deepseek-v2-lite", "deepseek-v2"],
    "Qwen3":   ["qwen3-30b-a3b",    "qwen3-235b-a22b"],
}
MODEL_TO_FAMILY = {m: f for f, ms in FAMILIES.items() for m in ms}


def tuple_idx(model, task):
    return TUPLES.index((model, task))


def alpha_idx(a):
    return ALPHAS.index(a)


def beta_idx(b):
    return BETAS.index(b)


# ---------------------------------------------------------------------------
# Load.
# ---------------------------------------------------------------------------
print(f"Loading {INPUT_PATH} ...")
data = np.load(INPUT_PATH, allow_pickle=True)
S = data["S"].astype(float)  # (64, 64, 5, 5)
print(f"  shape: {S.shape}")

# Coverage: count off-diagonal non-NaN cells at the (α=0, β=0) slice (any
# slice would work because all configs share the same (src, tgt) plan).
triu_i, triu_j = np.triu_indices(N_TUPLES, k=1)
pair_filled = ~np.isnan(S[triu_i, triu_j, 0, 0])
n_total = len(triu_i)
n_done = int(pair_filled.sum())
print(f"  coverage: {n_done}/{n_total} unordered cross-pairs "
      f"({100 * n_done / n_total:.1f}%)")


# ---------------------------------------------------------------------------
# Headline aggregates.
# ---------------------------------------------------------------------------
def nanmean(arr):
    arr = np.asarray(arr, dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if len(arr) else float("nan")


def collect_same_task(S_slice):
    """Return (within_family, cross_family) arrays of same-task S values."""
    wf, xf = [], []
    for t in TASKS:
        for mi, m1 in enumerate(MODELS):
            for mj, m2 in enumerate(MODELS):
                if mj <= mi:
                    continue
                i = tuple_idx(m1, t)
                j = tuple_idx(m2, t)
                s = S_slice[i, j]
                if np.isnan(s):
                    continue
                same_fam = (
                    MODEL_TO_FAMILY.get(m1) is not None
                    and MODEL_TO_FAMILY.get(m1) == MODEL_TO_FAMILY.get(m2)
                )
                (wf if same_fam else xf).append(s)
    return np.array(wf), np.array(xf)


def collect_within_model_cross_task(S_slice):
    """Mean S across the 28 pairs (t1, t2 distinct) for each model."""
    out = {}
    for m in MODELS:
        vals = []
        for ti, t1 in enumerate(TASKS):
            for tj, t2 in enumerate(TASKS):
                if tj <= ti:
                    continue
                i = tuple_idx(m, t1)
                j = tuple_idx(m, t2)
                s = S_slice[i, j]
                if not np.isnan(s):
                    vals.append(s)
        out[m] = (np.array(vals).mean() if vals else float("nan"),
                  len(vals))
    return out


def collect_per_task_cms(S_slice):
    """For each task, mean S over all cross-model same-task pairs."""
    out = {}
    for t in TASKS:
        vals = []
        for mi, m1 in enumerate(MODELS):
            for mj, m2 in enumerate(MODELS):
                if mj <= mi:
                    continue
                i = tuple_idx(m1, t)
                j = tuple_idx(m2, t)
                s = S_slice[i, j]
                if not np.isnan(s):
                    vals.append(s)
        out[t] = (np.array(vals).mean() if vals else float("nan"),
                  len(vals))
    return out


def collect_global_cms_cmd(S_slice):
    """Global CMS (same-task) and CMD (different-task) cross-model means."""
    cms, cmd = [], []
    for ti, t1 in enumerate(TASKS):
        for tj, t2 in enumerate(TASKS):
            for mi, m1 in enumerate(MODELS):
                for mj, m2 in enumerate(MODELS):
                    if mj <= mi:
                        continue
                    i = tuple_idx(m1, t1)
                    j = tuple_idx(m2, t2)
                    s = S_slice[i, j]
                    if np.isnan(s):
                        continue
                    (cms if ti == tj else cmd).append(s)
    return np.array(cms), np.array(cmd)


# ---------------------------------------------------------------------------
# Text summary.
# ---------------------------------------------------------------------------
lines = []
lines.append("=== Alpha-Beta Sweep Analysis ===")
lines.append(f"S.shape = {S.shape}   "
             f"models = {len(MODELS)}, tasks = {len(TASKS)}, "
             f"|TUPLES| = {N_TUPLES}, α = {ALPHAS}, β = {BETAS}")
lines.append(f"Coverage: {n_done}/{n_total} unordered cross-pairs "
             f"({100 * n_done / n_total:.1f}%)")
lines.append("")

# Pick three canonical slices.
canonical = [
    (0.0, 0.5, "pure feature"),
    (0.5, 0.5, "balanced"),
    (1.0, 0.5, "pure structure"),
]

for alpha, beta, label in canonical:
    a_i, b_i = alpha_idx(alpha), beta_idx(beta)
    sl = S[:, :, a_i, b_i]

    wf, xf = collect_same_task(sl)
    wm = collect_within_model_cross_task(sl)
    cms_per_t = collect_per_task_cms(sl)
    cms_all, cmd_all = collect_global_cms_cmd(sl)

    lines.append(f"--- α = {alpha}, β = {beta}   ({label}) ---")
    lines.append(f"  Within-family same-task  (n={len(wf):>3d}): "
                 f"mean = {wf.mean():.4f}, median = {np.median(wf):.4f}, "
                 f"min = {wf.min():.4f}, max = {wf.max():.4f}")
    lines.append(f"  Cross-family same-task  (n={len(xf):>3d}): "
                 f"mean = {xf.mean():.4f}, median = {np.median(xf):.4f}, "
                 f"min = {xf.min():.4f}, max = {xf.max():.4f}")
    lines.append(f"  Within-family − cross-family same-task gap: "
                 f"{wf.mean() - xf.mean():+.4f}")
    lines.append("")
    lines.append(f"  Global CMS (n={len(cms_all)}) = {cms_all.mean():.4f}")
    lines.append(f"  Global CMD (n={len(cmd_all)}) = {cmd_all.mean():.4f}")
    lines.append(f"  CMS − CMD                       = "
                 f"{cms_all.mean() - cmd_all.mean():+.4f}")
    lines.append("")
    lines.append("  Per-task CMS:")
    for t in TASKS:
        v, n = cms_per_t[t]
        lines.append(f"    {t:<12s}  CMS = {v:.4f}  (n = {n})")
    lines.append("")
    lines.append("  Per-model WM (within-model cross-task):")
    for m in MODELS:
        v, n = wm[m]
        lines.append(f"    {m:<22s}  WM = {v:.4f}  (n = {n})")
    lines.append("")

# Within-family pair × task tables.
lines.append("=== Within-family same-task table (α=0, β=0.5) ===")
sl = S[:, :, 0, beta_idx(0.5)]
header = ["family", "pair"] + TASKS + ["mean"]
lines.append("  " + " | ".join(f"{h:<14s}" for h in header))
lines.append("  " + "-" * (16 * len(header)))
for fam, (m1, m2) in FAMILIES.items():
    row = [fam, f"{m1[:6]}↔{m2[:6]}"]
    vals = []
    for t in TASKS:
        i = tuple_idx(m1, t)
        j = tuple_idx(m2, t)
        s = sl[i, j]
        row.append("NaN" if np.isnan(s) else f"{s:.4f}")
        if not np.isnan(s):
            vals.append(s)
    row.append(f"{np.mean(vals):.4f}" if vals else "NaN")
    lines.append("  " + " | ".join(f"{c:<14s}" for c in row))
lines.append("")

# Alpha-axis: how the global CMS - CMD shifts as α varies (at β=0.5).
lines.append("=== α-axis at β=0.5 (global CMS − CMD) ===")
for alpha in ALPHAS:
    sl = S[:, :, alpha_idx(alpha), beta_idx(0.5)]
    cms, cmd = collect_global_cms_cmd(sl)
    wf, xf = collect_same_task(sl)
    lines.append(f"  α = {alpha:.2f}  CMS = {cms.mean():.4f}  "
                 f"CMD = {cmd.mean():.4f}  "
                 f"gap = {cms.mean() - cmd.mean():+.4f}  "
                 f"within-fam = {wf.mean():.4f}  "
                 f"cross-fam = {xf.mean():.4f}")
lines.append("")

summary_text = "\n".join(lines)
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(summary_text + "\n")
print("\n" + summary_text)


# ---------------------------------------------------------------------------
# Plots.
# ---------------------------------------------------------------------------
def draw_heatmap(ax, S_slice, title, show_labels=True, fontsize=6):
    """Draw a 64x64 heatmap with model-block gridlines + within-family same-
    task cells highlighted with red boxes."""
    im = ax.imshow(S_slice, cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_title(title, fontsize=12)
    if show_labels:
        labels = [f"{m}/{t}" for m, t in TUPLES]
        ax.set_xticks(range(N_TUPLES))
        ax.set_yticks(range(N_TUPLES))
        ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
        ax.set_yticklabels(labels, fontsize=fontsize)
    else:
        ax.set_xticks([])
        ax.set_yticks([])

    # Model-block gridlines every 8 cells.
    for k in range(1, len(MODELS)):
        ax.axhline(k * len(TASKS) - 0.5, color="white", linewidth=0.5, alpha=0.5)
        ax.axvline(k * len(TASKS) - 0.5, color="white", linewidth=0.5, alpha=0.5)

    # Within-family same-task cells (red boxes).
    for fam, (m1, m2) in FAMILIES.items():
        for t in TASKS:
            i = tuple_idx(m1, t)
            j = tuple_idx(m2, t)
            for (yy, xx) in [(i, j), (j, i)]:
                ax.add_patch(plt.Rectangle((xx - 0.5, yy - 0.5), 1, 1,
                                           fill=False, edgecolor="red",
                                           linewidth=1.0))
    return im


# 1. Three canonical 64x64 heatmaps (separate files).
for alpha, beta, label, fname in [
    (0.0, 0.5, "α = 0 (pure feature), β = 0.5",   "heatmap_pure_feature.png"),
    (0.5, 0.5, "α = 0.5 (balanced), β = 0.5",     "heatmap_balanced.png"),
    (1.0, 0.5, "α = 1 (pure structure), β = 0.5", "heatmap_pure_structure.png"),
]:
    fig, ax = plt.subplots(figsize=(22, 20))
    sl = S[:, :, alpha_idx(alpha), beta_idx(beta)]
    im = draw_heatmap(ax, sl, label, show_labels=True)
    fig.colorbar(im, ax=ax, shrink=0.6)
    plt.tight_layout()
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")

# 2. 5 x 5 mini-heatmap grid over (α, β).
fig, axes = plt.subplots(len(ALPHAS), len(BETAS), figsize=(20, 20))
for ai, alpha in enumerate(ALPHAS):
    for bi, beta in enumerate(BETAS):
        ax = axes[ai, bi]
        sl = S[:, :, ai, bi]
        ax.imshow(sl, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"α = {alpha}, β = {beta}", fontsize=9)
        # Highlight within-family same-task cells.
        for fam, (m1, m2) in FAMILIES.items():
            for t in TASKS:
                i = tuple_idx(m1, t)
                j = tuple_idx(m2, t)
                for (yy, xx) in [(i, j), (j, i)]:
                    ax.add_patch(plt.Rectangle((xx - 0.5, yy - 0.5), 1, 1,
                                               fill=False, edgecolor="red",
                                               linewidth=0.5))
plt.suptitle("S landscape across (α, β)   "
             "[red = within-family same-task pairs]",
             fontsize=14, y=1.02)
plt.tight_layout()
out = os.path.join(OUT_DIR, "heatmap_alpha_beta_grid.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"  saved {out}")

# 3. Within-family pairs across tasks (bar chart, α=0).
fig, axes = plt.subplots(1, len(FAMILIES), figsize=(18, 5), sharey=True)
sl = S[:, :, 0, beta_idx(0.5)]
for fi, (fam, (m1, m2)) in enumerate(FAMILIES.items()):
    ax = axes[fi]
    vals = []
    for t in TASKS:
        i = tuple_idx(m1, t)
        j = tuple_idx(m2, t)
        vals.append(sl[i, j])
    bars = ax.bar(range(len(TASKS)), vals, color="#4C72B0")
    ax.set_title(f"{fam}\n{m1} ↔ {m2}", fontsize=11)
    ax.set_xticks(range(len(TASKS)))
    ax.set_xticklabels(TASKS, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("S (α=0, β=0.5)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3, axis="y")
    for k, v in enumerate(vals):
        if not np.isnan(v):
            ax.text(k, v + 0.02, f"{v:.2f}", ha="center", fontsize=8)
plt.tight_layout()
out = os.path.join(OUT_DIR, "within_family_pairs.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"  saved {out}")

# 4. Per-task CMS (α=0, β=0.5).
sl = S[:, :, 0, beta_idx(0.5)]
cms_per_t = collect_per_task_cms(sl)
fig, ax = plt.subplots(figsize=(10, 5))
ts = list(cms_per_t.keys())
vs = [cms_per_t[t][0] for t in ts]
ns = [cms_per_t[t][1] for t in ts]
bars = ax.bar(ts, vs, color="#55A868")
ax.set_ylabel("CMS = mean cross-model same-task S  (α=0, β=0.5)")
ax.set_title("Per-task cross-model alignment (CMS) at α=0")
ax.set_xticks(range(len(ts)))
ax.set_xticklabels(ts, rotation=45, ha="right")
ax.set_ylim(0, 1.05)
ax.grid(alpha=0.3, axis="y")
for k, (v, n) in enumerate(zip(vs, ns)):
    if not np.isnan(v):
        ax.text(k, v + 0.02, f"{v:.2f}\n(n={n})",
                ha="center", fontsize=8)
plt.tight_layout()
out = os.path.join(OUT_DIR, "cms_per_task.png")
plt.savefig(out, dpi=120, bbox_inches="tight")
plt.close()
print(f"  saved {out}")

print(f"\nAll outputs in: {OUT_DIR}")
