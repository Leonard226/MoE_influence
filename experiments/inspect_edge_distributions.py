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
  - PDF: edge_weights_<task>.pdf      (per-model log-histogram, 8 panels +
                                        Q-threshold vertical markers).
  - PDF: outgoing_mass_<task>.pdf     (per-model log-histogram, 8 panels).
  - PDF: edge_weights_overlay_<task>.pdf
                                       (all models in one panel, on log-density
                                        axes, easy cross-model comparison).

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
    """One panel per model, log-x histogram of the chosen quantity.

    Shared y-axis (fraction of edges per bin) so peak heights compare
    directly across models -- narrow / heavy-tailed distributions stand
    out as taller bars. Per-panel x-axis is clipped to the model's own
    data range so there's no empty space at the ends. Single restrained
    colour, vertical model labels, no figure title; the red dashed lines
    on the edges-ridge mark each model's per-graph Q-quantile thresholds
    (Q in {0.9, 0.99, 0.999}).
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
        fname = f"edge_weights_ridge_{task}.pdf"
    else:
        get_data = lambda r: r["out_mass"][r["out_mass"] > 0]
        xlabel = "Outgoing strength per expert (log-scale)"
        fname = f"outgoing_mass_ridge_{task}.pdf"

    arrays = [(r["model"], get_data(r)) for r in rows]
    arrays = [(m, a) for m, a in arrays if len(a) > 0]
    if not arrays:
        return out_dir / fname

    # Per-panel bins and fractions; keep the y_max so we can share the y-axis.
    N_BINS = 70
    panel_data: list[tuple[str, np.ndarray, np.ndarray, float, float]] = []
    y_max = 0.0
    for model, data in arrays:
        lo, hi = float(data.min()), float(data.max())
        bins = np.geomspace(lo, hi, N_BINS)
        counts, _ = np.histogram(data, bins=bins)
        frac = counts / counts.sum() if counts.sum() else counts.astype(float)
        panel_data.append((model, bins, frac, lo, hi))
        if frac.size:
            y_max = max(y_max, float(frac.max()))
    # Small headroom above the tallest peak.
    y_max_disp = y_max * 1.08 if y_max > 0 else 1.0

    # Single restrained colour across all panels: model identity is encoded
    # by the vertical row label, not by hue.
    BAR_COLOR = "#3a6d8c"   # muted slate-blue
    EDGE_COLOR = "#1f3a4d"

    fig, axes = plt.subplots(len(panel_data), 1,
                             figsize=(11, 1.55 * len(panel_data) + 0.9),
                             sharey=True)
    if len(panel_data) == 1:
        axes = [axes]

    for ax, (model, bins, frac, lo, hi) in zip(axes, panel_data):
        # Stepped histogram, filled. ax.stairs handles non-uniform log bins.
        ax.stairs(frac, edges=bins, fill=True,
                  facecolor=BAR_COLOR, edgecolor=EDGE_COLOR,
                  linewidth=0.6, alpha=0.95)

        ax.set_xscale("log")
        ax.set_xlim(lo, hi)            # clip to the model's own data range
        ax.set_ylim(0, y_max_disp)
        ax.tick_params(axis="x", which="major", labelsize=11)
        ax.tick_params(axis="x", which="minor", labelsize=0, length=2)
        ax.tick_params(axis="y", which="major", labelsize=10)
        ax.set_ylabel(model, rotation=90, ha="center", va="center",
                      fontsize=13, labelpad=10)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.grid(axis="y", linestyle="--", linewidth=0.4, alpha=0.45)

        # Per-graph Q-quantile thresholds on the edges ridge only.
        if kind == "edges":
            r = next(rr for rr in rows if rr["model"] == model)
            for Q, t in r["thresholds"].items():
                if t > 0 and lo <= t <= hi:
                    ax.axvline(t, color="#c0392b", linestyle="--",
                               linewidth=0.9, alpha=0.7)

    axes[-1].set_xlabel(xlabel, fontsize=14, labelpad=6)
    # Shared y-axis meaning, written once on the left.
    fig.supylabel("Fraction of edges per bin", fontsize=14, x=0.012)

    fig.tight_layout(rect=(0.02, 0, 1, 1))
    out_path = out_dir / fname
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_overlay(rows: list[dict], kind: str, task: str,
                  out_dir: Path, dpi: int) -> Path:
    """All-models overlay: filled stepped histograms with transparency,
    z-ordered so narrower (higher-peak) distributions sit on top of wider
    ones. Log-scale x-axis, raw values on ticks."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if not r.get("missing")]
    fig, ax = plt.subplots(figsize=(14, 8))

    # Stable model→color map: each model keeps its color regardless of plot order.
    base_colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))
    model_color = {r["model"]: c for r, c in zip(rows, base_colors)}

    # Data extraction.
    if kind == "edges":
        get_data = lambda r: r["edges"]
        xlabel = "|W|  (forward, nonzero edges only; log-scale)"
        title = (f"Forward edge-weight distribution: overlay across 8 models   "
                 f"(task = {task})")
        fname = f"edge_weights_overlay_{task}.pdf"
    else:
        get_data = lambda r: r["out_mass"][r["out_mass"] > 0]
        xlabel = "outgoing mass per expert  (log-scale)"
        title = (f"Per-expert outgoing-mass distribution: overlay across 8 models   "
                 f"(task = {task})")
        fname = f"outgoing_mass_overlay_{task}.pdf"

    arrays = [(r["model"], get_data(r)) for r in rows]
    arrays = [(m, a) for m, a in arrays if len(a) > 0]
    if not arrays:
        ax.text(0.5, 0.5, "(no data)", ha="center", transform=ax.transAxes)
        out_path = out_dir / fname
        fig.savefig(out_path, dpi=dpi)
        plt.close(fig)
        return out_path

    global_min = min(a.min() for _, a in arrays)
    global_max = max(a.max() for _, a in arrays)
    bins = np.geomspace(global_min, global_max, 100)

    # Pre-bin (fraction of edges per bin) so we can sort by peak height.
    # Narrowest (highest peak) is drawn LAST so it stays visible above wider
    # distributions. y-axis is fraction of edges per bin (not density), which
    # is directly interpretable on a log-scale x.
    binned = []
    for model, data in arrays:
        counts, _ = np.histogram(data, bins=bins)
        frac = counts / counts.sum() if counts.sum() > 0 else counts
        bin_centres = np.sqrt(bins[:-1] * bins[1:])
        binned.append((model, bin_centres, frac, frac.max()))
    binned.sort(key=lambda x: x[3])      # ascending: widest first

    for model, centres, frac, _ in binned:
        c = model_color[model]
        ax.fill_between(centres, 0, frac, step="mid",
                        facecolor=c, edgecolor=c,
                        linewidth=1.2, alpha=0.40,
                        label=model)

    ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel("fraction of edges per bin", fontsize=12)
    ax.tick_params(axis="both", which="major", labelsize=11)
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Legend ordered to match the canonical MODELS list (not the draw order),
    # so the user always sees the same models in the same legend position.
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(range(len(labels)),
                   key=lambda i: (MODELS.index(labels[i])
                                  if labels[i] in MODELS else 99))
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              loc="upper right", fontsize=12, framealpha=0.92)

    fig.tight_layout()
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
    p3 = _save_overlay(rows, "edges", args.task, out_dir, args.dpi)
    p4 = _save_overlay(rows, "mass", args.task, out_dir, args.dpi)
    p5 = _save_ridge(rows, "edges", args.task, out_dir, args.dpi)
    p6 = _save_ridge(rows, "mass", args.task, out_dir, args.dpi)
    print(f"\n  saved:")
    for p in (p1, p2, p3, p4, p5, p6):
        print(f"    {p}")


if __name__ == "__main__":
    main()
