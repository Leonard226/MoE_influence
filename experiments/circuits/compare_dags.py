"""Compare two DAGs across datasets to test cross-task persistence of edges.

Loads two DAGs produced by build_dag.py (e.g. dag_c4.pt and dag_math.pt) and
reports the cross-task persistence of routing-influence edges:

  - Spearman correlation of APS, ANS, mean across the two edge tensors.
  - Jaccard similarity of top-K edges at K ∈ {100, 1000, 10000}.
  - Named-edge consistency table for chains of interest.
  - Per-sender-layer stripe pattern comparison.

Usage:
    python experiments/circuits/compare_dags.py --datasets c4 math

Output:
    {result_path}/circuits/compare_{ds1}_vs_{ds2}.json
    stdout summary

CPU-only.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

try:
    from scipy.stats import spearmanr
except ImportError:
    sys.exit("Missing dependency. Install with:  pip install scipy")

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
art_dir = os.path.join(config["result_path"], "circuits")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--datasets", nargs=2, default=["c4", "math"],
                    metavar=("DS1", "DS2"),
                    help="Two dataset names to compare (reads dag_{ds}.pt each).")
parser.add_argument("--top-k", nargs="+", type=int, default=[100, 1000, 10000],
                    help="K values at which to compute Jaccard top-K overlap.")
args = parser.parse_args()
ds1, ds2 = args.datasets

N_LAYERS = 16
N_EXPERTS = 64

# Edges identified in prior analysis as candidates for circuit membership.
NAMED_EDGES = [
    (1,  9, 4, 14, "M1E9 → M4E14   (determiner chain, prior paper)"),
    (1,  9, 6,  4, "M1E9 → M6E4    (top APS promoter)"),
    (4, 14, 6,  4, "M4E14 → M6E4   (top inhibitor; mirrors above)"),
    (2, 30, 15,  7, "M2E30 → M15E7  (cap-init inhibition)"),
    (2, 30, 15, 10, "M2E30 → M15E10 (cap-init inhibition)"),
    (2, 30, 15, 13, "M2E30 → M15E13 (cap-init inhibition)"),
    (2, 30, 15, 21, "M2E30 → M15E21 (cap-init inhibition)"),
    (1,  9, 7, 61, "M1E9 → M7E61   (promoter)"),
    (4, 14, 9, 25, "M4E14 → M9E25  (promoter)"),
]


def load_dag(name):
    path = os.path.join(art_dir, f"dag_{name}.pt")
    print(f"Loading {path} ...")
    return torch.load(path, map_location="cpu")


def to_signed_array(t):
    return t.numpy().astype(np.float64).reshape(-1)


def jaccard_top_k(score_a, score_b, k):
    """Top-K by raw score (largest values). Returns intersection/union and counts."""
    if k > score_a.size:
        k = score_a.size
    top_a = set(np.argpartition(score_a, -k)[-k:].tolist())
    top_b = set(np.argpartition(score_b, -k)[-k:].tolist())
    inter = len(top_a & top_b)
    union = len(top_a | top_b)
    return inter / max(union, 1), inter, union


# ---- Load both DAGs ----
dag1 = load_dag(ds1)
dag2 = load_dag(ds2)

results = {"datasets": [ds1, ds2], "spearman": {}, "jaccard_top_k": {}, "named_edges": [], "stripe": {}}

# ---- Spearman correlations ----
print("\n========== Spearman correlations (per edge weight) ==========\n")
for key in ["APS", "ANS", "mean"]:
    a = to_signed_array(dag1[key])
    b = to_signed_array(dag2[key])
    # Filter to entries nonzero in either dataset (zeros are structurally invalid edges).
    mask = (a != 0) | (b != 0)
    if mask.sum() < 100:
        print(f"  {key}: too few overlapping edges ({mask.sum()})")
        continue
    rho, p = spearmanr(a[mask], b[mask])
    print(f"  {key:5s}: Spearman ρ = {rho:+.4f}  (p={p:.2e}, n={int(mask.sum())})")
    results["spearman"][key] = {"rho": float(rho), "p": float(p), "n": int(mask.sum())}

# ---- Jaccard top-K ----
print("\n========== Jaccard top-K overlap (per edge weight) ==========\n")

def rank_score_for(key, dag):
    """Score by which we rank edges, depending on weight semantics."""
    if key == "APS":
        return to_signed_array(dag[key])               # rank by largest positive
    elif key == "ANS":
        return -to_signed_array(dag[key])              # rank by largest |negative|
    elif key == "mean":
        return np.abs(to_signed_array(dag[key]))       # rank by magnitude (mixed sign)

for key in ["APS", "ANS", "mean"]:
    sa = rank_score_for(key, dag1)
    sb = rank_score_for(key, dag2)
    results["jaccard_top_k"][key] = {}
    print(f"  {key}:")
    for K in args.top_k:
        j, inter, union = jaccard_top_k(sa, sb, K)
        print(f"    top-{K:5d}: Jaccard = {j:.4f}  ({inter}/{union})")
        results["jaccard_top_k"][key][str(K)] = {"jaccard": j, "intersection": inter, "union": union}

# ---- Named edges ----
print("\n========== Named edges (consistency across datasets) ==========\n")
print(f"  {'edge':<60} {'metric':>5}   {ds1:>14}   {ds2:>14}")
for c, j, l, n, label in NAMED_EDGES:
    record = {"label": label, "c": c, "j": j, "l": l, "n": n, "values": {}}
    for key in ["APS", "ANS", "mean"]:
        v1 = float(dag1[key][c, j, l, n].item())
        v2 = float(dag2[key][c, j, l, n].item())
        print(f"  {label:<60} {key:>5}   {v1:+14.4e}   {v2:+14.4e}")
        record["values"][key] = {ds1: v1, ds2: v2}
    results["named_edges"].append(record)
    print()

# ---- Per-sender-layer stripe persistence (APS aggregate) ----
print("========== Per-sender-layer APS aggregate ==========\n")
sum1 = dag1["APS"].numpy().sum(axis=(1, 2, 3))
sum2 = dag2["APS"].numpy().sum(axis=(1, 2, 3))
print(f"  {'sender layer':<14} {ds1:>14} {ds2:>14}   ratio  rank{ds1}  rank{ds2}")
order1 = np.argsort(-sum1)
order2 = np.argsort(-sum2)
rank1 = np.empty_like(order1); rank1[order1] = np.arange(len(order1))
rank2 = np.empty_like(order2); rank2[order2] = np.arange(len(order2))

stripe = []
for c in range(N_LAYERS):
    ratio = sum2[c] / max(sum1[c], 1e-30)
    print(f"  M{c:<13d} {sum1[c]:14.4e} {sum2[c]:14.4e}  {ratio:6.2f}  {rank1[c]:>6d}  {rank2[c]:>6d}")
    stripe.append({
        "layer": c, ds1: float(sum1[c]), ds2: float(sum2[c]),
        f"rank_{ds1}": int(rank1[c]), f"rank_{ds2}": int(rank2[c]),
    })
results["stripe"] = stripe

# Per-sender-layer rank correlation: do the same layers dominate in both?
rho_stripe, _ = spearmanr(sum1, sum2)
print(f"\n  Sender-layer Spearman ρ = {rho_stripe:+.4f}  "
      f"(stripe pattern persistence; +1 = identical sender ranking)")
results["stripe_rank_spearman"] = float(rho_stripe)

# ---- Save ----
out_path = os.path.join(art_dir, f"compare_{ds1}_vs_{ds2}.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved {out_path}")

# ---- Headline summary ----
print("\n==================== Summary ====================\n")
print(f"  Datasets:  {ds1}  vs  {ds2}")
print(f"  Spearman ρ:  APS={results['spearman'].get('APS', {}).get('rho', float('nan')):+.4f}   "
      f"ANS={results['spearman'].get('ANS', {}).get('rho', float('nan')):+.4f}   "
      f"mean={results['spearman'].get('mean', {}).get('rho', float('nan')):+.4f}")
print(f"  Top-100 Jaccard: APS={results['jaccard_top_k']['APS']['100']['jaccard']:.4f}   "
      f"ANS={results['jaccard_top_k']['ANS']['100']['jaccard']:.4f}   "
      f"mean={results['jaccard_top_k']['mean']['100']['jaccard']:.4f}")
print(f"  Stripe rank ρ: {rho_stripe:+.4f}")
print()
print("Interpretation rules of thumb:")
print("  Spearman ρ > 0.7  → circuits broadly persist; supervisor's hypothesis confirmed.")
print("  ρ ∈ [0.3, 0.7]    → partial persistence; some circuits general, others task-specific.")
print("  ρ < 0.3           → circuits are mostly task-specific.")
