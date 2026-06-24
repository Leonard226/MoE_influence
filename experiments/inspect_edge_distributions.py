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

    log_edges = np.log10(edges) if len(edges) > 0 else np.array([])
    log_mass = np.log10(out_mass[out_mass > 0]) if (out_mass > 0).any() else np.array([])

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
        "log_edges_pct": (np.percentile(log_edges, [1, 5, 25, 50, 75, 95, 99]).tolist()
                          if len(log_edges) > 0 else []),
        "out_mass_zero_count": int((out_mass == 0).sum()),
        "out_mass_min": float(out_mass.min()),
        "out_mass_max": float(out_mass.max()),
        "out_mass_mean": float(out_mass.mean()),
        "out_mass_median": float(np.median(out_mass)),
        "log_mass_pct": (np.percentile(log_mass, [1, 5, 25, 50, 75, 95, 99]).tolist()
                         if len(log_mass) > 0 else []),
        "thresholds": thresholds,
        "log_edges": log_edges,                # for plotting
        "out_mass": out_mass,                  # for plotting
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
    """8-panel multi-model histogram. kind in {'edges', 'mass'}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if not r.get("missing")]
    fig, axes = plt.subplots(2, 4, figsize=(22, 10))
    axes = axes.flatten()

    for i, r in enumerate(rows[:8]):
        ax = axes[i]
        if kind == "edges":
            data = r["log_edges"]
            xlabel = "log10(|W|)  (forward, nonzero edges only)"
            title = (f"{r['model']}\nV={r['V']}, |E|={r['n_fwd_present']:,} "
                     f"({100*r['edge_density']:.1f}% density)")
        else:  # mass
            mass_nz = r["out_mass"][r["out_mass"] > 0]
            data = np.log10(mass_nz) if len(mass_nz) > 0 else np.array([])
            xlabel = "log10(outgoing mass per expert)"
            title = (f"{r['model']}\n"
                     f"zero-mass experts: {r['out_mass_zero_count']}/{r['V']}")
        if len(data) == 0:
            ax.set_title(title + "\n(no data)")
            ax.set_xlabel(xlabel)
            continue
        ax.hist(data, bins=80, color="steelblue", alpha=0.85)
        # Q-threshold markers on the edges panel.
        if kind == "edges":
            for Q, t in r["thresholds"].items():
                if t > 0:
                    ax.axvline(np.log10(t), color="red", linestyle="--",
                               linewidth=1.0, alpha=0.65,
                               label=f"Q={Q}: {t:.1e}")
            ax.legend(loc="upper left", fontsize=7, framealpha=0.85)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("count")
        ax.set_title(title, fontsize=10)
    for j in range(len(rows), 8):
        axes[j].axis("off")

    fig.suptitle(
        ("Forward |W| log10-histogram per model" if kind == "edges"
         else "Outgoing mass log10-histogram per model")
        + f"   (task = {task})",
        fontsize=14,
    )
    fig.tight_layout()
    out_path = out_dir / f"{'edge_weights' if kind == 'edges' else 'outgoing_mass'}_{task}.pdf"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _save_edges_overlay(rows: list[dict], task: str,
                        out_dir: Path, dpi: int) -> Path:
    """All-models edge-weight overlay on log-density axes."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [r for r in rows if not r.get("missing")]
    fig, ax = plt.subplots(figsize=(14, 8))
    colors = plt.cm.tab10(np.linspace(0, 1, len(rows)))
    for r, c in zip(rows, colors):
        if len(r["log_edges"]) == 0:
            continue
        ax.hist(r["log_edges"], bins=120, histtype="step", linewidth=1.6,
                color=c, label=r["model"], density=True, alpha=0.85)
    ax.set_xlabel("log10(|W|)  (forward, nonzero edges only)")
    ax.set_ylabel("density")
    ax.set_title(f"Forward edge-weight distribution: overlay across 8 models   "
                 f"(task = {task})", fontsize=13)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    out_path = out_dir / f"edge_weights_overlay_{task}.pdf"
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
    p3 = _save_edges_overlay(rows, args.task, out_dir, args.dpi)
    print(f"\n  saved:")
    print(f"    {p1}")
    print(f"    {p2}")
    print(f"    {p3}")


if __name__ == "__main__":
    main()
