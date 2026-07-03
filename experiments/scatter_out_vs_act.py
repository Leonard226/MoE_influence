"""Per-expert log-log scatter of out(v) vs act(v), colored by layer depth.

For each model on --task:
  - Plots all V experts as small points (color = fractional layer depth).
  - Overlays the identified Super Experts (Su-exact, --include-layers 0.75)
    as red circles.
  - Overlays the top-K experts by out(v) (our "Global Experts" set) as
    hollow black squares. Their intersection with the SE overlay is
    exactly the bold entries in tab:super-experts-c4.

Prints per-model rank correlations:
  - rho_S(out, act)          — raw Spearman
  - rho_S(in,  act)          — raw Spearman
  - rho_S(out, in)           — raw Spearman
  - rho_S(out, act | depth)  — partial Spearman controlling for layer

Usage:
    python experiments/scatter_out_vs_act.py
    python experiments/scatter_out_vs_act.py --task math --top-k-global 5
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    _config = yaml.safe_load(f)
CIRCUITS_DIR = Path(_config["result_path"]) / "circuits"
DEFAULT_OUT_DIR = CIRCUITS_DIR / "distribution_inspection"

MODELS = [
    "mixtral-8x7b", "mixtral-8x22b", "phi-3.5-moe",
    "deepseek-v2-lite", "olmoe",
    "qwen3-30b-a3b", "qwen3-235b-a22b", "deepseek-v2",
]

# Same convention as print_top_acts.py: model-absolute layer indices
# used by Su, translated to our DAG's dense-shifted range.
NUM_DENSE = {
    "mixtral-8x7b": 0, "mixtral-8x22b": 0,
    "phi-3.5-moe": 0,
    "deepseek-v2-lite": 1, "deepseek-v2": 1,
    "olmoe": 0,
    "qwen3-30b-a3b": 0, "qwen3-235b-a22b": 0,
}


def _load_features(model: str, task: str):
    """Return dict with flat np.float64 arrays: out, in, load, act, depth (+L, N).
    load(v) = n_tok(v) / mean_{n'} n_tok(l, n')  (per-layer mean-normalised)."""
    path = CIRCUITS_DIR / f"dag_{model}_{task}.pt"
    if not path.exists():
        return None
    dag = torch.load(path, weights_only=False, map_location="cpu")
    if "W_softmax" not in dag or "act" not in dag or "n_tokens_selected" not in dag:
        return None
    W = dag["W_softmax"].to(torch.float64)
    L, N, _, _ = W.shape
    s = torch.arange(L).view(-1, 1, 1, 1)
    r = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s < r).to(W.dtype)
    W_fwd = W * fwd
    out_LN = W_fwd.abs().sum(dim=(2, 3)).numpy()   # sender side
    in_LN = W_fwd.abs().sum(dim=(0, 1)).numpy()    # receiver side
    act_LN = dag["act"].to(torch.float64).numpy()
    n_tok = dag["n_tokens_selected"].to(torch.float64).numpy()  # [L, N]
    layer_mean = n_tok.mean(axis=1, keepdims=True).clip(min=1e-12)
    load_LN = n_tok / layer_mean                    # [L, N], mean = 1 per layer
    depth_LN = np.broadcast_to(np.arange(L)[:, None], (L, N)).astype(np.float64)
    return {"out": out_LN.reshape(-1), "in": in_LN.reshape(-1),
            "load": load_LN.reshape(-1),
            "act": act_LN.reshape(-1), "depth": depth_LN.reshape(-1),
            "L": L, "N": N, "act_LN": act_LN}


def _partial_spearman(x, y, z):
    """Partial Spearman(x, y | z) via classical partial-correlation formula
    on the rank-transformed variables (spearmanr already ranks internally)."""
    r_xy, _ = spearmanr(x, y)
    r_xz, _ = spearmanr(x, z)
    r_yz, _ = spearmanr(y, z)
    denom = np.sqrt(max(0.0, (1 - r_xz ** 2) * (1 - r_yz ** 2)))
    return float("nan") if denom == 0 else (r_xy - r_xz * r_yz) / denom


def _se_mask(model: str, act_LN: np.ndarray, include_layers_frac: float) -> np.ndarray:
    """Su-exact SE indicator on the flat expert grid.
    Matches print_top_acts.py: np.percentile linear interpolation + floor div,
    computed on the layer-filtered subset only. Returns bool array shape (V,)."""
    L, N = act_LN.shape
    num_dense = NUM_DENSE.get(model, 0)
    total_model_L = L + num_dense
    include_up_to_model = round(total_model_L * include_layers_frac)
    include_up_to_dag = max(0, min(L, include_up_to_model - num_dense))
    subset = act_LN[:include_up_to_dag].reshape(-1)
    if subset.size == 0:
        return np.zeros(L * N, dtype=bool)
    p_995 = np.percentile(subset, 99.5)
    top1_x_01 = np.max(subset) // 10
    thr = max(p_995, top1_x_01)
    mask_LN = np.zeros((L, N), dtype=bool)
    mask_LN[:include_up_to_dag] = act_LN[:include_up_to_dag] > thr
    return mask_LN.reshape(-1)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--x-axis", choices=["out", "in", "load"], default="out",
                   help="Which routing-graph strength to put on the x-axis "
                        "(default 'out'). Overlays: red circles = Super Experts, "
                        "black squares = top-K by the same x-axis metric.")
    p.add_argument("--include-layers", type=float, default=0.75,
                   help="Su's include_layers fraction for the SE overlay (default 0.75).")
    p.add_argument("--top-k-global", type=int, default=5,
                   help="Number of top experts (by the --x-axis metric) to "
                        "overlay as 'Top-K by <x-axis>' (default 5).")
    p.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- Collect features + correlations ---
    data = {}
    print(f"{'model':<18s} {'V':>6s}  {'rho(out,act)':>13s}  {'rho(in,act)':>13s}  "
          f"{'rho(out,in)':>13s}  {'rho(out,act|d)':>15s}  {'rho(in,act|d)':>14s}")
    print("-" * 102)
    for m in MODELS:
        r = _load_features(m, args.task)
        if r is None:
            print(f"{m:<18s}  MISSING")
            continue
        rho_out_act, _ = spearmanr(r["out"], r["act"])
        rho_in_act, _ = spearmanr(r["in"], r["act"])
        rho_out_in, _ = spearmanr(r["out"], r["in"])
        rho_partial_out = _partial_spearman(r["out"], r["act"], r["depth"])
        rho_partial_in = _partial_spearman(r["in"], r["act"], r["depth"])
        V = r["out"].size
        print(f"{m:<18s} {V:>6d}  {rho_out_act:>13.3f}  {rho_in_act:>13.3f}  "
              f"{rho_out_in:>13.3f}  {rho_partial_out:>15.3f}  {rho_partial_in:>14.3f}")
        data[m] = {**r, "rho_out_act": rho_out_act, "rho_partial": rho_partial_out}

    if not data:
        print("No models found; aborting.")
        return

    # --- Scatter grid ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10), constrained_layout=True)
    axes = axes.ravel()
    cmap = plt.get_cmap("viridis")

    x_key = args.x_axis
    x_label = rf"$\mathrm{{{x_key}}}(v)$"
    sc = None
    for i, (m, d) in enumerate(data.items()):
        ax = axes[i]
        x_v = d[x_key].copy()
        act_v = d["act"].copy()
        # Log axes: replace non-positives with (min positive)/10 so the point
        # is still visible at the axis floor.
        for arr in (x_v, act_v):
            pos = arr[arr > 0]
            floor = pos.min() / 10 if pos.size else 1e-12
            arr[arr <= 0] = floor
        depth_frac = d["depth"] / max(1, d["L"] - 1)
        sc = ax.scatter(x_v, act_v, c=depth_frac, cmap=cmap, vmin=0, vmax=1,
                        s=6, alpha=0.5, edgecolors="none")

        # Overlay: SE (Su-exact) as red circles
        se_mask = _se_mask(m, d["act_LN"], args.include_layers)
        if se_mask.any():
            ax.scatter(x_v[se_mask], act_v[se_mask],
                       facecolors="none", edgecolors="crimson",
                       s=60, linewidths=1.3,
                       label=f"Super Expert (n={se_mask.sum()})")

        # Overlay: top-K by the x-axis metric, as black squares
        top_g = np.argsort(-d[x_key])[:args.top_k_global]
        ax.scatter(x_v[top_g], act_v[top_g],
                   facecolors="none", edgecolors="black",
                   marker="s", s=80, linewidths=1.3,
                   label=f"Top-{args.top_k_global} by {x_key}")

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_title(m, fontsize=12)
        ax.set_xlabel(x_label)
        ax.set_ylabel(r"$\mathrm{act}(v)$")
        ax.grid(True, which="both", alpha=0.2)
        ax.legend(fontsize=8, loc="lower left", framealpha=0.85,
                  labelspacing=1.2, handletextpad=0.8, borderpad=0.6)

    for j in range(len(data), len(axes)):
        axes[j].set_visible(False)

    if sc is not None:
        cbar = fig.colorbar(sc, ax=axes.tolist(), shrink=1.0, aspect=40,
                            label="fractional layer depth")
        cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])

    out_path = out_dir / f"{args.x_axis}_vs_act_scatter_{args.task}.pdf"
    fig.savefig(out_path, dpi=200)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
