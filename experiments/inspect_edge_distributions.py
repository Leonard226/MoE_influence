"""Per-model diagnostic of |W| edge-weight and outgoing-mass distributions.

For each model on the chosen task:
  1. Load the DAG and extract |W| over forward edges.
  2. Compute outgoing-mass per expert (sum of |W| over outgoing forward edges).
  3. Compute the per-graph Q-quantile thresholds at Q in QUANTILES.

This answers questions like:
  - Are the per-graph Q-quantile thresholds keeping comparable edge density
    across models, or is the same Q producing radically different absolute
    thresholds (Mixtral keeps edges down to 0.013; Qwen keeps edges down to
    5e-4 at the same Q=0.9)?
  - How concentrated is the outgoing influence per expert? Heavy-tailed
    (super-experts) or balanced?
  - Are there experts with zero outgoing mass (i.e., never selected on the
    calibration corpus)?

Outputs (under {result_path}/circuits/distribution_inspection/):
  - console: per-model summary table.
  - PDF: edge_weights_<task>.pdf       (per-model log-histogram, 8 panels +
                                        Q-threshold vertical markers).
  - PDF: outgoing_mass_<task>.pdf      (per-model log-histogram, 8 panels).
  - PDF: edge_weights_ridge_<task>.pdf (one stacked panel per model, shared
                                        log-x axis, with Q-threshold markers).
  - PDF: outgoing_mass_ridge_<task>.pdf (same layout, outgoing strength).

Usage:
    python experiments/inspect_edge_distributions.py
    python experiments/inspect_edge_distributions.py --task math
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.run_alpha_beta_sweep import (  # noqa: E402
    MODELS, QUANTILES, EDGE_TENSOR,
)

with open(os.path.join(ROOT, "config.yaml")) as f:
    _config = yaml.safe_load(f)
CIRCUITS_DIR = Path(_config["result_path"]) / "circuits"
DEFAULT_OUT_DIR = CIRCUITS_DIR / "distribution_inspection"


def _compute_stats(model: str, task: str) -> dict:
    """Forward-edge |W| and per-expert outgoing-mass stats."""
    dag_path = CIRCUITS_DIR / f"dag_{model}_{task}.pt"
    if not dag_path.exists():
        return {"model": model, "task": task, "missing": True}
    dag = torch.load(dag_path, weights_only=False, map_location="cpu")
    W = dag[EDGE_TENSOR].float()
    L, N = W.shape[0], W.shape[1]
    V = L * N

    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    W_fwd = W * fwd.float()
    n_fwd_possible = int(fwd.sum())

    W_abs = W_fwd.abs().reshape(V, V).cpu().numpy().astype(np.float64)
    edges = W_abs[W_abs > 0]                  # nonzero forward edges only
    out_mass = W_abs.sum(axis=1)              # outgoing mass per expert (V,)

    # Q-quantile thresholds taken over the same set the sweep uses (nonzero forward).
    thresholds = {Q: float(np.quantile(edges, Q)) if len(edges) > 0 else 0.0
                  for Q in QUANTILES}

    return {
        "model": model, "task": task, "missing": False,
        "L": L, "N": N, "V": V,
        "n_fwd_possible": n_fwd_possible,
        "n_fwd_present": int(len(edges)),
        "edge_density": len(edges) / max(n_fwd_possible, 1),
        "edges_min": float(edges.min()) if len(edges) > 0 else 0.0,
        "edges_max": float(edges.max()) if len(edges) > 0 else 0.0,
        "edges_mean": float(edges.mean()) if len(edges) > 0 else 0.0,
        "edges_median": float(np.median(edges)) if len(edges) > 0 else 0.0,
        "edges_pct": (np.percentile(edges, [1, 5, 25, 50, 75, 95, 99]).tolist()
                      if len(edges) > 0 else []),
        "out_mass_zero_count": int((out_mass == 0).sum()),
        "out_mass_min": float(out_mass.min()),
        "out_mass_max": float(out_mass.max()),
        "out_mass_mean": float(out_mass.mean()),
        "out_mass_median": float(np.median(out_mass)),
        "out_mass_pct": (np.percentile(out_mass[out_mass > 0],
                                        [1, 5, 25, 50, 75, 95, 99]).tolist()
                         if (out_mass > 0).any() else []),
        "thresholds": thresholds,
        "edges": edges,                        # raw |W| (plot on log axis)
        "out_mass": out_mass,                  # raw per-expert mass
    }


def _print_summary(rows: list[dict]) -> None:
    rows = [r for r in rows if not r.get("missing")]
    if not rows:
        print("(no models loaded)")
        return
    print("\n" + "=" * 140)
    print("  Per-model |W| (forward, nonzero) and outgoing-mass summary")
    print("=" * 140)
    print(f"  {'model':<18s}  {'V':>6s}  {'|E| present':>11s}  {'density':>8s}  "
          f"{'|W| min':>10s} {'|W| med':>10s} {'|W| max':>10s}  "
          f"{'mass med':>10s} {'mass max':>10s}  "
          f"{'thr Q=.9':>10s} {'thr Q=.99':>10s} {'thr Q=.999':>10s}")
    print("  " + "-" * 138)
    for r in rows:
        thr = r["thresholds"]
        print(f"  {r['model']:<18s}  {r['V']:>6d}  {r['n_fwd_present']:>11,}  "
              f"{100 * r['edge_density']:>7.2f}%  "
              f"{r['edges_min']:>10.2e} {r['edges_median']:>10.2e} {r['edges_max']:>10.2e}  "
              f"{r['out_mass_median']:>10.2e} {r['out_mass_max']:>10.2e}  "
              f"{thr[0.9]:>10.2e} {thr[0.99]:>10.2e} {thr[0.999]:>10.2e}")
    print("\n  Edge density = (present forward edges) / (possible forward edges).")
    print("  |W| medians span orders of magnitude across models — the same Q quantile")
    print("  therefore corresponds to very different absolute thresholds.")


def _save_per_model_panels(rows: list[dict], kind: str, task: str,
                           out_dir: Path, dpi: int) -> Path:
    """8-panel multi-model histogram. kind in {'edges', 'mass'}.
    Uses raw values with a log-scale x-axis (ticks like 1e-6, 1e-4, ...)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if not r.get("missing")]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, r in enumerate(rows[:8]):
        ax = axes[i]
        if kind == "edges":
            data = r["edges"]
            xlabel = "|W|  (forward, nonzero edges only; log-scale)"
            title = (f"{r['model']}\nV={r['V']}, |E|={r['n_fwd_present']:,} "
                     f"({100*r['edge_density']:.1f}% density)")
        else:  # mass
            data = r["out_mass"][r["out_mass"] > 0]
            xlabel = "outgoing mass per expert  (log-scale)"
            title = (f"{r['model']}\n"
                     f"zero-mass experts: {r['out_mass_zero_count']}/{r['V']}")
        if len(data) == 0:
            ax.set_title(title + "\n(no data)")
            ax.set_xlabel(xlabel)
            continue
        bins = np.geomspace(data.min(), data.max(), 80)
        ax.hist(data, bins=bins, color="steelblue", alpha=0.85)
        ax.set_xscale("log")
        # Q-threshold markers on the edges panel (in raw |W| units now).
        if kind == "edges":
            for Q, t in r["thresholds"].items():
                if t > 0:
                    ax.axvline(t, color="red", linestyle="--",
                               linewidth=1.0, alpha=0.65,
                               label=f"Q={Q}: {t:.1e}")
            ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.set_title(title, fontsize=10)
    for j in range(len(rows), 8):
        axes[j].axis("off")

    fig.suptitle(
        ("Forward |W| distribution per model (log x-axis)" if kind == "edges"
         else "Outgoing mass per expert per model (log x-axis)")
        + f"   (task = {task})",
        fontsize=14,
    )
    fig.tight_layout()
    out_path = out_dir / f"{'edge_weights' if kind == 'edges' else 'outgoing_mass'}_{task}.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_ridge(rows: list[dict], kind: str, task: str,
                out_dir: Path, dpi: int) -> Path:
    """Two-column ridge plot: one row per model; left column = linear y,
    right column = log y. Both columns plot the same histogram with the
    same shared log-x; the two y-scales let the reader read both the bulk
    shape (linear) and the heavy tail / rare super-experts (log) without
    flipping figures. Vertical model labels, no figure title. The red
    dashed lines on the edges-ridge mark each model's per-graph Q-quantile
    thresholds (Q in {0.9, 0.99, 0.999}).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if not r.get("missing")]
    if not rows:
        return out_dir / "ridge_empty.pdf"
    rows = sorted(rows, key=lambda r: MODELS.index(r["model"])
                                       if r["model"] in MODELS else 99)

    if kind == "edges":
        get_data = lambda r: r["edges"]
        xlabel = "Edge weight magnitude (log-scale)"
        ylabel = "Fraction of edges per bin"
        fname = f"edge_weights_ridge_{task}.pdf"
    else:
        get_data = lambda r: r["out_mass"][r["out_mass"] > 0]
        xlabel = "Outgoing strength per expert (log-scale)"
        ylabel = "Fraction of experts per bin"
        fname = f"outgoing_mass_ridge_{task}.pdf"

    arrays = [(r["model"], get_data(r)) for r in rows]
    arrays = [(m, a) for m, a in arrays if len(a) > 0]
    if not arrays:
        return out_dir / fname

    # Shared bins on the global data range so a vertical at x is the same
    # value across every panel.
    N_BINS = 100
    global_min = min(float(a.min()) for _, a in arrays)
    global_max = max(float(a.max()) for _, a in arrays)
    bins = np.geomspace(global_min, global_max, N_BINS)

    panel_data: list[tuple[str, np.ndarray]] = []
    y_max = 0.0
    y_min_positive = float("inf")    # smallest non-zero fraction across all models
    for model, data in arrays:
        counts, _ = np.histogram(data, bins=bins)
        frac = counts / counts.sum() if counts.sum() else counts.astype(float)
        panel_data.append((model, frac))
        if frac.size:
            y_max = max(y_max, float(frac.max()))
            pos = frac[frac > 0]
            if pos.size:
                y_min_positive = min(y_min_positive, float(pos.min()))
    y_max_disp = y_max * 1.08 if y_max > 0 else 1.0

    # Single, conservative colour across panels. Model identity is encoded
    # by the vertical row label, not by hue. tab:blue is the matplotlib
    # default and reads cleanly in print.
    BAR_COLOR = "#1f77b4"
    EDGE_COLOR = "#0e3d63"

    n_rows = len(panel_data)
    lo_y = y_min_positive * 0.5 if np.isfinite(y_min_positive) else 1e-5
    fig, axes = plt.subplots(n_rows, 2,
                             figsize=(13, 1.55 * n_rows + 1.0),
                             sharex=True, sharey=False)
    if n_rows == 1:
        axes = axes.reshape(1, 2)

    # Per-model row thresholds for the edges-ridge red markers.
    model_thresholds: dict[str, dict] = {
        rr["model"]: rr["thresholds"] for rr in rows
    }

    for ri, (model, frac) in enumerate(panel_data):
        for ci, scale in enumerate(("linear", "log")):
            ax = axes[ri, ci]
            ax.stairs(frac, edges=bins, fill=True,
                      facecolor=BAR_COLOR, edgecolor=EDGE_COLOR,
                      linewidth=0.6, alpha=0.95)
            ax.set_xscale("log")
            ax.set_xlim(global_min, global_max)
            if scale == "linear":
                ax.set_ylim(0, y_max_disp)
            else:
                ax.set_yscale("log")
                ax.set_ylim(lo_y, y_max_disp)
            ax.tick_params(axis="x", which="major", labelsize=10)
            ax.tick_params(axis="x", which="minor", labelsize=0, length=2)
            ax.tick_params(axis="y", which="major", labelsize=9)
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45)
            # Model name as ylabel on the RIGHT side of the log-y panel.
            if ci == 1:
                ax.yaxis.set_label_position("right")
                ax.set_ylabel(model, rotation=90, ha="center", va="center",
                              fontsize=12, labelpad=10, fontweight="bold")
            # Per-graph Q-quantile threshold markers on edges-ridge only.
            if kind == "edges":
                for Q, t in model_thresholds[model].items():
                    if t > 0:
                        ax.axvline(t, color="#c0392b", linestyle="--",
                                   linewidth=0.9, alpha=0.7)

    # Column headers (top row), bold.
    axes[0, 0].set_title("linear scale", fontsize=12, pad=4, fontweight="bold")
    axes[0, 1].set_title("log scale",    fontsize=12, pad=4, fontweight="bold")

    # Bottom-row x-axis label on both columns (shared x), bold.
    for ci in (0, 1):
        axes[-1, ci].set_xlabel(xlabel, fontsize=13, labelpad=6,
                                fontweight="bold")

    # Shared y-axis meaning label, moved rightward (closer to the panels) and
    # bold for emphasis.
    fig.supylabel(ylabel, fontsize=13, x=0.028, fontweight="bold")

    # Reserve right-side margin for the per-row model labels.
    fig.tight_layout(rect=(0.035, 0, 0.96, 1))
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path




def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", default="c4")
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--dpi", type=int, default=180)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for model in MODELS:
        print(f"  loading {model}/{args.task} ...", end=" ", flush=True)
        r = _compute_stats(model, args.task)
        rows.append(r)
        if r.get("missing"):
            print("MISSING")
        else:
            print(f"V={r['V']}, |E|={r['n_fwd_present']:,} "
                  f"({100*r['edge_density']:.2f}% density)")

    _print_summary(rows)

    p1 = _save_per_model_panels(rows, "edges", args.task, out_dir, args.dpi)
    p2 = _save_per_model_panels(rows, "mass", args.task, out_dir, args.dpi)
    p5 = _save_ridge(rows, "edges", args.task, out_dir, args.dpi)
    p6 = _save_ridge(rows, "mass", args.task, out_dir, args.dpi)
    print(f"\n  saved:")
    for p in (p1, p2, p5, p6):
        print(f"    {p}")


if __name__ == "__main__":
    main()
