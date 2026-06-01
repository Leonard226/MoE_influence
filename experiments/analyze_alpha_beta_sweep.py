"""Headline analysis of the alpha × Q quantile FGW sweep.

Reads S_full.npz produced by aggregate_alpha_beta_sweep.py
(shape (64, 64, len(ALPHAS), len(QUANTILES))) and writes to
{result_path}/circuits/alpha_beta_sweep/analysis/:

    summary.txt
        Coverage, global aggregates per (α, Q), per-task CMS, per-model WM,
        within-family pair tables.
    heatmap_a{α}_Q{Q}.png    (9 PNGs, one per (α, Q) cell)
        Per-task 2×4 grid of 8×8 model×model heatmaps, within-family model
        pairs boxed in red, cell values overlaid.
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
    MODELS, TASKS, TUPLES, N_TUPLES, ALPHAS, QUANTILES, FIXED_BETA,
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
    "Mixtral":  ["mixtral-8x7b",     "mixtral-8x22b"],
    "DeepSeek": ["deepseek-v2-lite", "deepseek-v2"],
    "Qwen3":    ["qwen3-30b-a3b",    "qwen3-235b-a22b"],
}
MODEL_TO_FAMILY = {m: f for f, ms in FAMILIES.items() for m in ms}


def tuple_idx(model, task):
    return TUPLES.index((model, task))


def alpha_idx(a):
    return ALPHAS.index(a)


def q_idx(q):
    return QUANTILES.index(q)


# ---------------------------------------------------------------------------
# Load.
# ---------------------------------------------------------------------------
print(f"Loading {INPUT_PATH} ...")
data = np.load(INPUT_PATH, allow_pickle=True)
S = data["S"].astype(float)   # (64, 64, n_alpha, n_q)
print(f"  shape: {S.shape}")
assert S.shape == (N_TUPLES, N_TUPLES, len(ALPHAS), len(QUANTILES)), \
    f"unexpected shape {S.shape}"

# Coverage: count off-diagonal non-NaN cells at the (α=0, Q=0.9) slice.
triu_i, triu_j = np.triu_indices(N_TUPLES, k=1)
pair_filled = ~np.isnan(S[triu_i, triu_j, 0, 0])
n_total = len(triu_i)
n_done = int(pair_filled.sum())
print(f"  coverage: {n_done}/{n_total} unordered cross-pairs "
      f"({100 * n_done / n_total:.1f}%)")


# ---------------------------------------------------------------------------
# Aggregates — take any (α, Q) 2D slice, slice-agnostic.
# ---------------------------------------------------------------------------
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
        out[m] = (np.array(vals).mean() if vals else float("nan"), len(vals))
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
        out[t] = (np.array(vals).mean() if vals else float("nan"), len(vals))
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
ALPHA_LABEL = {0.0: "pure feature", 0.5: "balanced", 1.0: "pure structure"}

lines = []
lines.append("=== α × Q Quantile FGW Sweep Analysis ===")
lines.append(f"S.shape = {S.shape}   "
             f"models = {len(MODELS)}, tasks = {len(TASKS)}, "
             f"|TUPLES| = {N_TUPLES}, α = {ALPHAS}, "
             f"Q = {QUANTILES}, β = {FIXED_BETA} (fixed)")
lines.append(f"Coverage: {n_done}/{n_total} unordered cross-pairs "
             f"({100 * n_done / n_total:.1f}%)")
lines.append("")

# Per-(α, Q) detailed block.
for alpha in ALPHAS:
    for Q in QUANTILES:
        a_i, q_i = alpha_idx(alpha), q_idx(Q)
        sl = S[:, :, a_i, q_i]

        wf, xf = collect_same_task(sl)
        wm = collect_within_model_cross_task(sl)
        cms_per_t = collect_per_task_cms(sl)
        cms_all, cmd_all = collect_global_cms_cmd(sl)

        label = ALPHA_LABEL.get(alpha, f"α={alpha}")
        lines.append(f"--- α = {alpha}, Q = {Q}   ({label}) ---")
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

# Within-family pair × task table at α=0, Q highest (most-discriminative under sparsification).
ref_Q = QUANTILES[-1]
ref_alpha = 0.0
lines.append(f"=== Within-family same-task table (α={ref_alpha}, Q={ref_Q}) ===")
sl_ref = S[:, :, alpha_idx(ref_alpha), q_idx(ref_Q)]
header = ["family", "pair"] + TASKS + ["mean"]
lines.append("  " + " | ".join(f"{h:<14s}" for h in header))
lines.append("  " + "-" * (16 * len(header)))
for fam, (m1, m2) in FAMILIES.items():
    row = [fam, f"{m1[:6]}↔{m2[:6]}"]
    vals = []
    for t in TASKS:
        i = tuple_idx(m1, t)
        j = tuple_idx(m2, t)
        s = sl_ref[i, j]
        row.append("NaN" if np.isnan(s) else f"{s:.4f}")
        if not np.isnan(s):
            vals.append(s)
    row.append(f"{np.mean(vals):.4f}" if vals else "NaN")
    lines.append("  " + " | ".join(f"{c:<14s}" for c in row))
lines.append("")

# α × Q landscape — compact CMS / CMD / gap grid.
lines.append("=== α × Q landscape (global CMS, CMD, gap, within-fam, cross-fam) ===")
lines.append("  α \\ Q  ||  " + "  ".join(f"Q={Q:<6.3g}" for Q in QUANTILES))
lines.append("  " + "-" * (12 + 14 * len(QUANTILES)))
for alpha in ALPHAS:
    cells_gap = []
    cells_wf  = []
    cells_xf  = []
    cells_cms = []
    cells_cmd = []
    for Q in QUANTILES:
        sl = S[:, :, alpha_idx(alpha), q_idx(Q)]
        cms, cmd = collect_global_cms_cmd(sl)
        wf, xf = collect_same_task(sl)
        cells_cms.append(cms.mean())
        cells_cmd.append(cmd.mean())
        cells_gap.append(cms.mean() - cmd.mean())
        cells_wf.append(wf.mean())
        cells_xf.append(xf.mean())
    lines.append(f"  α = {alpha:.2f}")
    lines.append(f"    CMS         : " + "  ".join(f"{v:>8.4f}" for v in cells_cms))
    lines.append(f"    CMD         : " + "  ".join(f"{v:>8.4f}" for v in cells_cmd))
    lines.append(f"    CMS-CMD gap : " + "  ".join(f"{v:>+8.4f}" for v in cells_gap))
    lines.append(f"    within-fam  : " + "  ".join(f"{v:>8.4f}" for v in cells_wf))
    lines.append(f"    cross-fam   : " + "  ".join(f"{v:>8.4f}" for v in cells_xf))
    lines.append("")

summary_text = "\n".join(lines)
with open(os.path.join(OUT_DIR, "summary.txt"), "w") as f:
    f.write(summary_text + "\n")
print("\n" + summary_text)


# ---------------------------------------------------------------------------
# Plots: nine per-(α, Q) heatmaps, each a 2×4 grid of per-task 8×8 model heatmaps.
# ---------------------------------------------------------------------------
def draw_per_task_heatmaps(S_slice, fig_title, fname):
    """One 8×8 model-model heatmap per task, laid out in a 2×4 grid.
    Within-family model pairs (Mixtral 7B/22B, DSL/V2, Qwen 30B/235B) are
    boxed in red in every subplot."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    im = None
    for ti, t in enumerate(TASKS):
        ax = axes[ti]
        idx = [tuple_idx(m, t) for m in MODELS]
        sub = S_slice[np.ix_(idx, idx)]
        im = ax.imshow(sub, cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_title(t, fontsize=12, fontweight="bold")
        ax.set_xticks(range(len(MODELS)))
        ax.set_yticks(range(len(MODELS)))
        ax.set_xticklabels(MODELS, rotation=45, ha="right", fontsize=8)
        # Only label y-axis on the leftmost column; rows are identical across.
        if ti % 4 == 0:
            ax.set_yticklabels(MODELS, fontsize=8)
        else:
            ax.set_yticklabels([])
        # Cell value overlays.
        for i in range(len(MODELS)):
            for j in range(len(MODELS)):
                v = sub[i, j]
                if not np.isnan(v):
                    ax.text(j, i, f"{v:.2f}",
                            ha="center", va="center", fontsize=7,
                            color="white" if v < 0.6 else "black")
        # Within-family model pairs (red boxes).
        for m1, m2 in FAMILIES.values():
            i = MODELS.index(m1)
            j = MODELS.index(m2)
            for (yy, xx) in [(i, j), (j, i)]:
                ax.add_patch(plt.Rectangle((xx - 0.5, yy - 0.5), 1, 1,
                                           fill=False, edgecolor="red",
                                           linewidth=1.5))
    fig.suptitle(fig_title, fontsize=15, fontweight="bold", y=1.00)
    fig.colorbar(im, ax=axes.tolist(), shrink=0.7, location="right",
                 label="S (FGW similarity)")
    out = os.path.join(OUT_DIR, fname)
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"  saved {out}")


def _q_tag(q):
    """0.9 -> '0.9', 0.99 -> '0.99', 0.999 -> '0.999' (consistent labels)."""
    s = f"{q:g}"
    return s


for alpha in ALPHAS:
    for Q in QUANTILES:
        sl = S[:, :, alpha_idx(alpha), q_idx(Q)]
        label_name = ALPHA_LABEL.get(alpha, f"α={alpha}")
        title = (f"α = {alpha} ({label_name}),  Q = {Q},  "
                 f"β = {FIXED_BETA} (fixed)")
        fname = f"heatmap_a{alpha:g}_Q{_q_tag(Q)}.png"
        draw_per_task_heatmaps(sl, title, fname)

print(f"\nAll outputs in: {OUT_DIR}")
