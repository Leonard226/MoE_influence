"""Leiden communities on a DAG produced by build_dag.py.

Tests whether the routing influence DAG has modular, multi-layer expert
communities. Builds two directed weighted graphs (APS for promotion, |ANS|
for inhibition), runs Leiden, compares modularity Q to a degree-preserving
null, and reports per-community statistics with attention to layer span.

Usage:
    python experiments/circuits/analyze_communities.py --dataset c4
    python experiments/circuits/analyze_communities.py --dataset math

Output:
    {result_path}/circuits/communities_{dataset}.json
    stdout summary

CPU-only.

Dependencies: python-igraph, leidenalg
    pip install python-igraph leidenalg
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

try:
    import igraph as ig
    import leidenalg
except ImportError as e:
    sys.exit(
        "Missing dependency. Install with:\n"
        "    pip install python-igraph leidenalg\n"
        f"Original error: {e}"
    )

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
art_dir = os.path.join(config["result_path"], "circuits")
os.makedirs(art_dir, exist_ok=True)

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="c4",
                    help="Which DAG to analyze (default: c4). Reads dag_{dataset}.pt.")
parser.add_argument("--n-null-trials", type=int, default=3,
                    help="Number of degree-preserving null trials (default: 3).")
args = parser.parse_args()

N_LAYERS = 16
N_EXPERTS = 64
N_NODES = N_LAYERS * N_EXPERTS

N_NULL_TRIALS = args.n_null_trials
# Top-N communities to print/save in detail.
TOP_N_COMMUNITIES = 10
# Random seed for null reproducibility.
RNG_SEED = 0


# ---- Load DAG ----
dag_path = os.path.join(art_dir, f"dag_{args.dataset}.pt")
print(f"Loading {dag_path} ...", flush=True)
data = torch.load(dag_path, map_location="cpu")
APS  = data["APS"].numpy()                              # [c, j, l, n]
ANS  = data["ANS"].numpy()
print(f"  APS  range: [{APS.min():+.4f}, {APS.max():+.4f}]")
print(f"  ANS  range: [{ANS.min():+.4f}, {ANS.max():+.4f}]")


# ---- Helpers ----
def expert_id(c, j):
    return c * N_EXPERTS + j


def expert_name(idx):
    c = idx // N_EXPERTS
    j = idx % N_EXPERTS
    return f"M{c}E{j}"


def build_igraph(W4, name):
    """Directed weighted graph from a 4D non-negative tensor [c, j, l, n].

    Edges with zero weight are dropped (saves memory; Leiden ignores them anyway).
    """
    nz_mask = W4 > 0
    cs, js, ls, ns = np.nonzero(nz_mask)
    sources = (cs * N_EXPERTS + js).astype(np.int64)
    targets = (ls * N_EXPERTS + ns).astype(np.int64)
    weights = W4[nz_mask].astype(np.float64)

    g = ig.Graph(directed=True, n=N_NODES)
    g.add_edges(np.column_stack((sources, targets)).tolist())
    g.es["weight"] = weights.tolist()
    print(f"  {name} graph: {g.vcount()} nodes, {g.ecount()} edges, "
          f"weight in [{weights.min():.4e}, {weights.max():.4e}]")
    return g


def run_leiden(g, name):
    """Standard directed-modularity Leiden on weighted graph."""
    t0 = time.time()
    partition = leidenalg.find_partition(
        g,
        leidenalg.ModularityVertexPartition,
        weights="weight",
        n_iterations=10,
        seed=RNG_SEED,
    )
    Q = partition.modularity
    membership = np.array(partition.membership)
    n_comm = len(set(membership))
    print(f"  {name}: Q = {Q:.4f},  {n_comm} communities,  "
          f"({time.time() - t0:.1f}s)")
    return Q, membership, n_comm


def null_modularity(g, name, n_trials=N_NULL_TRIALS):
    """Degree-preserving rewiring null. Q averaged over n_trials randomizations.

    Rewires edges (preserves in/out degree per vertex), then permutes weights so
    they're dissociated from the original topology. Runs Leiden on each null.
    """
    print(f"  Running {n_trials} null trials for {name} ...")
    Qs = []
    rng = np.random.default_rng(RNG_SEED)
    for trial in range(n_trials):
        g_null = g.copy()
        # Degree-preserving edge swap. n = 10*ecount is plenty for full mixing.
        g_null.rewire(n=10 * g.ecount(), mode="simple")
        # Permute weights so they're not tied to original topology.
        permuted = rng.permutation(g.es["weight"]).tolist()
        g_null.es["weight"] = permuted

        partition_null = leidenalg.find_partition(
            g_null,
            leidenalg.ModularityVertexPartition,
            weights="weight",
            n_iterations=5,
            seed=RNG_SEED + trial + 1,
        )
        Qs.append(partition_null.modularity)
        print(f"    trial {trial+1}: Q_null = {Qs[-1]:.4f}")
    return float(np.mean(Qs)), float(np.std(Qs))


def summarize_communities(membership, weight_4d, name, top_n=TOP_N_COMMUNITIES):
    """Print and return summary stats for the top-N largest communities.

    For each community: size, layer span, layers, top members, and average
    in-community edge weight.
    """
    sizes = np.bincount(membership)
    sorted_ids = np.argsort(sizes)[::-1]

    print(f"\n  Top {top_n} {name} communities:")
    print(f"    {'rank':>4}  {'id':>4}  {'size':>5}  {'layers':>20}  {'span':>4}")

    out = []
    for rank, comm_id in enumerate(sorted_ids[:top_n]):
        members = np.where(membership == comm_id)[0]
        if len(members) == 0:
            continue
        layers = members // N_EXPERTS
        layer_set = sorted(set(layers.tolist()))
        span = max(layers) - min(layers) if len(layers) > 1 else 0

        # Compute average edge weight WITHIN this community.
        member_set = set(members.tolist())
        within_weights = []
        for src in members:
            c = src // N_EXPERTS
            j = src % N_EXPERTS
            for l in range(c + 1, N_LAYERS):
                for n in range(N_EXPERTS):
                    if (l * N_EXPERTS + n) in member_set:
                        w = float(weight_4d[c, j, l, n])
                        if w > 0:
                            within_weights.append(w)
        avg_within = float(np.mean(within_weights)) if within_weights else 0.0

        layer_str = ",".join(str(l) for l in layer_set[:6])
        if len(layer_set) > 6:
            layer_str += ",..."
        print(f"    {rank+1:>4}  {comm_id:>4}  {len(members):>5}  "
              f"{layer_str:>20}  {span:>4}  "
              f"avg_within_w={avg_within:.4f}")

        member_names = [expert_name(int(m)) for m in members]
        # Show first 12 members.
        head = ", ".join(member_names[:12])
        more = "" if len(member_names) <= 12 else f" + {len(member_names) - 12} more"
        print(f"      members: {head}{more}")

        out.append({
            "rank": rank + 1,
            "comm_id": int(comm_id),
            "size": int(len(members)),
            "layers": [int(l) for l in layer_set],
            "layer_span": int(span),
            "avg_within_weight": avg_within,
            "members": [(int(m // N_EXPERTS), int(m % N_EXPERTS)) for m in members.tolist()],
        })
    return out


# ============================================================
# Main analysis
# ============================================================
print("\n--- Building graphs ---")
g_APS = build_igraph(APS, "APS")
g_ANS = build_igraph(np.abs(ANS), "|ANS|")

print("\n--- APS communities (promotion structure) ---")
Q_APS, mem_APS, n_APS = run_leiden(g_APS, "APS")
Qn_APS_mean, Qn_APS_std = null_modularity(g_APS, "APS")
print(f"  APS:  Q = {Q_APS:.4f},  Q_null = {Qn_APS_mean:.4f} ± {Qn_APS_std:.4f},  "
      f"Q − Q_null = {Q_APS - Qn_APS_mean:+.4f}")
APS_top = summarize_communities(mem_APS, APS, "APS")

print("\n--- |ANS| communities (inhibition structure) ---")
Q_ANS, mem_ANS, n_ANS = run_leiden(g_ANS, "|ANS|")
Qn_ANS_mean, Qn_ANS_std = null_modularity(g_ANS, "|ANS|")
print(f"  |ANS|:  Q = {Q_ANS:.4f},  Q_null = {Qn_ANS_mean:.4f} ± {Qn_ANS_std:.4f},  "
      f"Q − Q_null = {Q_ANS - Qn_ANS_mean:+.4f}")
ANS_top = summarize_communities(mem_ANS, np.abs(ANS), "|ANS|")

# ---- Save ----
out = {
    "APS": {
        "Q": Q_APS,
        "Q_null_mean": Qn_APS_mean,
        "Q_null_std": Qn_APS_std,
        "Q_minus_Q_null": Q_APS - Qn_APS_mean,
        "n_communities": n_APS,
        "membership": mem_APS.tolist(),
        "top_communities": APS_top,
    },
    "ANS": {
        "Q": Q_ANS,
        "Q_null_mean": Qn_ANS_mean,
        "Q_null_std": Qn_ANS_std,
        "Q_minus_Q_null": Q_ANS - Qn_ANS_mean,
        "n_communities": n_ANS,
        "membership": mem_ANS.tolist(),
        "top_communities": ANS_top,
    },
    "meta": {
        "n_layers": N_LAYERS,
        "n_experts": N_EXPERTS,
        "n_nodes": N_NODES,
        "source": dag_path,
        "n_null_trials": N_NULL_TRIALS,
        "rng_seed": RNG_SEED,
    },
}
out_path = os.path.join(art_dir, f"communities_{args.dataset}.json")
with open(out_path, "w") as f:
    json.dump(out, f, indent=2)
print(f"\nSaved {out_path}")

# ---- Final summary ----
print("\n==================== Summary ====================\n")
print(f"APS   (promotion):  Q = {Q_APS:.4f}, Q_null = {Qn_APS_mean:.4f}  "
      f"(diff = {Q_APS - Qn_APS_mean:+.4f}),  {n_APS} communities")
print(f"|ANS| (inhibition): Q = {Q_ANS:.4f}, Q_null = {Qn_ANS_mean:.4f}  "
      f"(diff = {Q_ANS - Qn_ANS_mean:+.4f}),  {n_ANS} communities")
print()
print("Interpretation:")
print("  Q − Q_null > 0.05 (rule of thumb): real modular structure beyond degree.")
print("  Q − Q_null ≈ 0:                    structure is just degree noise.")
print("  Layer span > 0:                    community spans multiple layers (good).")
print("  Layer span = 0 (single layer):     trivial single-layer community.")
