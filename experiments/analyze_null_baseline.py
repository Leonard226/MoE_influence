"""Combine null-baseline FGW results with the headline sweep to test whether
trained-trained within-family similarity is meaningfully above the trained-
vs-random null floor.

Reads:
  - Headline sweep   : alpha_beta_sweep_<suffix>/S_full_with_act.npz
                       (trained-trained S values for all 64 model/task pairs)
  - Null baseline    : null_baseline_<suffix>/null_S.csv
                       (4212 rows: trained-vs-random + random-vs-random pairs
                       across 4 comparison types, 3 alphas, 3 Q values)

For each (α, Q) cell, computes statistics for SIX comparison types:
  - trained_within   trained-trained within-family (from headline, same-task)
  - trained_cross    trained-trained cross-family (from headline, same-task)
  - tr_vs_rnd_same   trained vs random of the same architecture (null)
  - tr_vs_rnd_cross  trained vs random of a different architecture (null)
  - rnd_vs_rnd_cross random vs random, different architectures (null)
  - rnd_vs_rnd_same  random vs random, same architecture (seed variance)

Key derived numbers per (α, Q):
  - Δ_within_null  = mean(trained_within) − mean(tr_vs_rnd_cross)
  - Cohen-d-like effect size of trained_within above the null
  - Whether trained_within > 95th percentile of tr_vs_rnd_cross (one-sided)

Outputs (under the null subdir, same suffix as the headline sweep):
  - null_analysis_summary.txt   per-(α,Q) tabulated stats + effect sizes
  - null_violins.pdf            per-(α,Q) violin plot, 6 categories side by side
  - null_overlay_alpha0p5_Q0p999.pdf
                                 the headline-cell-as-paper-figure: overlaid
                                 histograms for one carefully chosen (α, Q)

Usage:
  python experiments/analyze_null_baseline.py \\
      --structural-mode conn --beta 0 --gamma 0.5
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.run_alpha_beta_sweep import (   # noqa: E402
    MODELS, TASKS, TUPLES, ALPHAS, QUANTILES, FIXED_BETA,
)

with open(os.path.join(ROOT, "config.yaml")) as f:
    _config = yaml.safe_load(f)
CIRCUITS_DIR = Path(_config["result_path"]) / "circuits"

FAMILIES = {
    "Mixtral":  ["mixtral-8x7b", "mixtral-8x22b"],
    "DeepSeek": ["deepseek-v2-lite", "deepseek-v2"],
    "Qwen3":    ["qwen3-30b-a3b", "qwen3-235b-a22b"],
}
MODEL_TO_FAMILY = {m: f for f, ms in FAMILIES.items() for m in ms}

NULL_TASK = "c4"     # null baseline was run on c4 only

# Three meaningful categories for the trained-vs-random comparison:
#   * trained x trained (within)   — same-family, same-task; FROM HEADLINE TENSOR
#                                    across all 8 tasks  (n = 3 within-family pair-archs × 8 tasks = 24)
#   * trained x trained (cross)    — different-family, same-task; FROM HEADLINE TENSOR
#                                    across all 8 tasks  (n = 25 cross-family pair-archs × 8 tasks = 200)
#   * trained x random (same arch) — null hypothesis: does training imprint
#                                    structure over random init of the same arch?
#                                    FROM NULL CSV (c4 only, 8 archs × 3 seeds = 24)
CATEGORIES = [
    # (key, csv string or "headline:...", display label, colour)
    ("trained_within", "headline:within",
        "trained × trained\n(within)",        "#1f6cb0"),
    ("trained_cross",  "headline:cross",
        "trained × trained\n(cross)",         "#5dade2"),
    ("tr_vs_rnd_same", "trained_vs_random_same",
        "trained × random\n(same arch)",      "#e67e22"),
]


# -------------------- path resolution --------------------------------------
def _sweep_suffix(structural_mode: str, beta: float, gamma: float) -> str:
    parts = ["logact", "logload"]                 # always log_max in our pipeline
    if structural_mode == "local":
        parts.append("local")
    elif structural_mode == "conn":
        parts.append("conn")
    if beta != FIXED_BETA:
        parts.append(f"b{beta:g}")
    if structural_mode == "conn" and gamma != 1.0:
        parts.append(f"g{gamma:g}")
    return "_".join(parts)


def _null_suffix(structural_mode: str, beta: float, gamma: float) -> str:
    parts: list[str] = []
    if structural_mode == "local":
        parts.append("local")
    elif structural_mode == "conn":
        parts.append("conn")
    if beta != FIXED_BETA:
        parts.append(f"b{beta:g}")
    if structural_mode == "conn" and gamma != 1.0:
        parts.append(f"g{gamma:g}")
    return "_".join(parts)


# -------------------- data loading -----------------------------------------
def _load_headline(structural_mode: str, beta: float, gamma: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (S, alphas, quantiles)."""
    suffix = _sweep_suffix(structural_mode, beta, gamma)
    path = CIRCUITS_DIR / f"alpha_beta_sweep_{suffix}" / "S_full_with_act.npz"
    print(f"  headline sweep S tensor: {path}")
    if not path.exists():
        raise FileNotFoundError(f"headline sweep S tensor not found: {path}")
    d = np.load(path, allow_pickle=True)
    return d["S"].astype(np.float64), d["alphas"], d["quantiles"]


def _load_null(structural_mode: str, beta: float, gamma: float) -> list[dict]:
    """Returns list of row-dicts from null_S.csv."""
    suffix = _null_suffix(structural_mode, beta, gamma)
    subdir = f"null_baseline_{suffix}" if suffix else "null_baseline"
    path = CIRCUITS_DIR / subdir / "null_S.csv"
    print(f"  null baseline CSV:        {path}")
    if not path.exists():
        raise FileNotFoundError(
            f"null baseline CSV not found: {path}\n"
            "Did you run `run_null_baseline.py --merge` with matching flags?"
        )
    rows: list[dict] = []
    with open(path) as f:
        for row in csv.DictReader(f):
            try:
                rows.append({
                    "comparison": row["comparison"],
                    "alpha": float(row["alpha"]),
                    "Q": float(row["Q"]),
                    "S": float(row["S"]),
                })
            except ValueError:
                # Skip any rows where S is NaN/empty (FGW solver failures).
                continue
    return rows


def _trained_pairs(S: np.ndarray, alpha_idx: int, Q_idx: int
                   ) -> tuple[np.ndarray, np.ndarray]:
    """For one (α, Q) cell, return (within_family, cross_family) same-task
    pair arrays from the headline tensor, AGGREGATED OVER ALL 8 TASKS.
    Same-task means both endpoints are on the same task d; we then average
    across d. Within-family: n = 3 within-family arch-pairs × 8 tasks = 24.
    Cross-family : n = 25 cross-family arch-pairs × 8 tasks = 200."""
    within, cross = [], []
    for task in TASKS:
        for mi, m1 in enumerate(MODELS):
            for mj, m2 in enumerate(MODELS):
                if mj <= mi:
                    continue
                ti = TUPLES.index((m1, task))
                tj = TUPLES.index((m2, task))
                v = S[ti, tj, alpha_idx, Q_idx]
                if np.isnan(v) or v < 0:    # -1 = FGW solver failure sentinel
                    continue
                fam1 = MODEL_TO_FAMILY.get(m1)
                fam2 = MODEL_TO_FAMILY.get(m2)
                same = (fam1 is not None and fam1 == fam2)
                (within if same else cross).append(float(v))
    return np.array(within), np.array(cross)


def _collect_cell(S, headline_alphas, headline_Qs, alpha, Q,
                  null_rows) -> dict[str, np.ndarray]:
    """Three category arrays for one (α, Q) cell."""
    ai = int(np.argmin(np.abs(np.asarray(headline_alphas) - alpha)))
    qi = int(np.argmin(np.abs(np.asarray(headline_Qs) - Q)))
    within, cross = _trained_pairs(S, ai, qi)
    out: dict[str, np.ndarray] = {
        "trained_within": within,
        "trained_cross":  cross,
    }
    # Pull trained × random (same arch) from the null CSV. Null was run on
    # c4 only, so this is c4 only by construction.
    null_csv_key = "trained_vs_random_same"
    vals = [r["S"] for r in null_rows
            if abs(r["alpha"] - alpha) < 1e-9
            and abs(r["Q"] - Q) < 1e-9
            and r["comparison"] == null_csv_key]
    out["tr_vs_rnd_same"] = np.array(vals, dtype=np.float64)
    return out


# -------------------- console / text summary -------------------------------
def _print_and_save_summary(S, headline_alphas, headline_Qs, null_rows,
                            out_path: Path) -> Path:
    lines: list[str] = []
    def w(s: str = "") -> None:
        print(s); lines.append(s)

    w("=" * 92)
    w(f"Null-baseline analysis  (task = {NULL_TASK})")
    w("=" * 92)

    for alpha in ALPHAS:
        for Q in QUANTILES:
            cell = _collect_cell(S, headline_alphas, headline_Qs, alpha, Q,
                                  null_rows)
            w("")
            w(f"--- α = {alpha}, Q = {Q} ---")
            w(f"  {'category':<22s} {'n':>5s}  {'mean':>8s}  {'std':>8s}  "
              f"{'median':>8s}  {'p05':>8s}  {'p95':>8s}")
            for key, _csv, _lbl, _col in CATEGORIES:
                v = cell[key]
                if len(v) == 0:
                    w(f"  {key:<22s} {0:>5d}  (no data)")
                    continue
                w(f"  {key:<22s} {len(v):>5d}  {v.mean():>8.4f}  {v.std():>8.4f}  "
                  f"{np.median(v):>8.4f}  {np.percentile(v, 5):>8.4f}  "
                  f"{np.percentile(v, 95):>8.4f}")

            # Effect-size summary: does training imprint structural signature
            # on top of architecture? Compare trained × trained (within-family,
            # all tasks) against trained × random (same arch, c4-only null).
            tw = cell["trained_within"]
            null = cell["tr_vs_rnd_same"]
            if len(tw) > 0 and len(null) > 0:
                delta = float(tw.mean() - null.mean())
                pooled = float(np.sqrt(0.5 * (tw.var() + null.var())))
                d = delta / pooled if pooled > 1e-12 else float("nan")
                p95_null = float(np.percentile(null, 95))
                frac_above = float((tw > p95_null).mean())
                w(f"  >> trained × trained (within) vs trained × random "
                  f"(same arch):")
                w(f"     Δmean = {delta:+.4f},  Cohen d = {d:+.2f},  "
                  f"fraction above null p95 = {100*frac_above:.0f}%")

    w("")
    w("=" * 92)
    out_path.write_text("\n".join(lines))
    print(f"\nSaved: {out_path}")
    return out_path


# -------------------- plotting ---------------------------------------------
def _save_violins(S, headline_alphas, headline_Qs, null_rows,
                  out_dir: Path, dpi: int) -> Path:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_a, n_q = len(ALPHAS), len(QUANTILES)
    fig, axes = plt.subplots(n_a, n_q, figsize=(5 * n_q, 4 * n_a),
                              squeeze=False, sharey=True)

    for ai, alpha in enumerate(ALPHAS):
        for qi, Q in enumerate(QUANTILES):
            ax = axes[ai, qi]
            cell = _collect_cell(S, headline_alphas, headline_Qs, alpha, Q,
                                  null_rows)
            data = []
            colours = []
            labels = []
            for key, _csv, label, colour in CATEGORIES:
                v = cell[key]
                if len(v) == 0:
                    v = np.array([np.nan])
                data.append(v)
                colours.append(colour)
                labels.append(label)
            # violinplot rejects empty arrays; replace any all-nan placeholders.
            data_safe = [v if not np.all(np.isnan(v)) else np.zeros(1)
                         for v in data]
            parts = ax.violinplot(data_safe, showmeans=True, showmedians=False,
                                  widths=0.8)
            for body, c in zip(parts["bodies"], colours):
                body.set_facecolor(c)
                body.set_edgecolor("black")
                body.set_alpha(0.78)
                body.set_linewidth(0.7)
            for key in ("cmeans", "cbars", "cmins", "cmaxes"):
                if key in parts:
                    parts[key].set_color("black")
                    parts[key].set_linewidth(0.9)

            ax.set_xticks(range(1, len(CATEGORIES) + 1))
            ax.set_xticklabels(labels, fontsize=8, rotation=25, ha="right")
            ax.set_title(f"α = {alpha},  Q = {Q}", fontsize=12)
            if qi == 0:
                ax.set_ylabel(r"FGW similarity  $\mathcal{S}_\alpha$", fontsize=11)
            ax.grid(alpha=0.18, axis="y")
            ax.set_ylim(-0.02, 1.03)

    fig.suptitle(
        "Trained × Trained vs. Trained × Random null per (α, Q)\n"
        "trained × trained averaged over all 8 tasks; "
        f"trained × random on '{NULL_TASK}' only (where the null was sampled)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / "null_violins.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


def _save_focus_overlay(S, headline_alphas, headline_Qs, null_rows,
                        out_dir: Path, dpi: int,
                        focus_alpha: float = 0.5,
                        focus_Q: float = 0.999) -> Path:
    """One carefully-chosen (α, Q) cell, all 6 distributions as overlaid
    KDE/histograms. Intended as the paper headline figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cell = _collect_cell(S, headline_alphas, headline_Qs,
                          focus_alpha, focus_Q, null_rows)

    fig, ax = plt.subplots(figsize=(11, 7))
    bins = np.linspace(0.0, 1.0, 60)
    centres = (bins[:-1] + bins[1:]) / 2

    for key, _csv, label, colour in CATEGORIES:
        v = cell[key]
        if len(v) == 0:
            continue
        counts, _ = np.histogram(v, bins=bins)
        frac = counts / counts.sum() if counts.sum() > 0 else counts
        ax.fill_between(centres, 0, frac, step="mid",
                        facecolor=colour, edgecolor=colour,
                        linewidth=1.2, alpha=0.45,
                        label=f"{label.replace(chr(10), ' ')}  (n={len(v)}, "
                              f"mean={v.mean():.3f})")

    ax.set_xlabel(r"FGW similarity  $\mathcal{S}_\alpha$", fontsize=13)
    ax.set_ylabel("fraction of pairs per bin", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_title(
        f"α = {focus_alpha},  Q = {focus_Q}\n"
        f"trained × trained averaged over all 8 tasks; "
        f"trained × random on '{NULL_TASK}' only",
        fontsize=13,
    )
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    a_str = f"{focus_alpha:g}".replace(".", "p")
    q_str = f"{focus_Q:g}".replace(".", "p")
    out_path = out_dir / f"null_overlay_alpha{a_str}_Q{q_str}.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"Saved: {out_path}")
    return out_path


# -------------------- entry point ------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--structural-mode", default="path",
                        choices=["path", "local", "conn"],
                        help="Match the structural-mode used by the sweep.")
    parser.add_argument("--beta", type=float, default=FIXED_BETA,
                        help=f"Match the --beta used by the sweep "
                             f"(default {FIXED_BETA}).")
    parser.add_argument("--gamma", type=float, default=1.0,
                        help="Match the --gamma used by the sweep (default 1.0).")
    parser.add_argument("--focus-alpha", type=float, default=0.5,
                        help="α cell to highlight in the overlay figure.")
    parser.add_argument("--focus-Q", type=float, default=0.999,
                        help="Q cell to highlight in the overlay figure.")
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    S, headline_alphas, headline_Qs = _load_headline(
        args.structural_mode, args.beta, args.gamma)
    null_rows = _load_null(args.structural_mode, args.beta, args.gamma)
    print(f"Loaded headline S tensor of shape {S.shape}")
    print(f"Loaded {len(null_rows)} null rows from CSV\n")

    null_subdir = f"null_baseline_{_null_suffix(args.structural_mode, args.beta, args.gamma)}"
    out_dir = CIRCUITS_DIR / null_subdir / "analysis"
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_and_save_summary(S, headline_alphas, headline_Qs, null_rows,
                             out_dir / "null_analysis_summary.txt")
    _save_violins(S, headline_alphas, headline_Qs, null_rows, out_dir, args.dpi)
    _save_focus_overlay(S, headline_alphas, headline_Qs, null_rows,
                         out_dir, args.dpi,
                         focus_alpha=args.focus_alpha, focus_Q=args.focus_Q)


if __name__ == "__main__":
    main()
