"""Discover circuit candidates from a DAG via graph algorithms.

Pipeline: load dag_<dataset>.pt, sparsify under several edge-weight choices,
apply chain enumeration / community detection / fan-out detection / cross-task
differential analysis, and write a JSON of candidates ready for ablation
validation.

Usage:
    python experiments/circuits/discover_circuits.py --dataset c4
    python experiments/circuits/discover_circuits.py --dataset c4 --compare-with math

Outputs (written under {result_path}/circuits/):
    candidates_<dataset>.json     — chains, communities, fan-outs.
    differential_<a>_vs_<b>.json  — task-specific edges (only with --compare-with).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

from experiments.circuits.graph_utils import (
    chain_str,
    dag_to_igraph,
    differential_edges,
    expert_name,
    extract_chains,
    extract_communities,
    extract_fan_outs,
)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
art_dir = os.path.join(config["result_path"], "circuits")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", default="c4", help="Dataset to discover circuits on (reads dag_<dataset>.pt).")
parser.add_argument("--compare-with", default=None, help="Optional second dataset for differential-edge analysis.")
parser.add_argument("--top-K", type=int, default=5, help="Per-sender top-K for sparsification (default: 5).")
parser.add_argument("--chain-min-length", type=int, default=2, help="Min number of edges in returned chains (default: 2).")
parser.add_argument("--chain-max-length", type=int, default=10, help="Max number of edges in returned chains (default: 4).")
parser.add_argument("--max-paths", type=int, default=100_000_000, help="Number of paths to enumerate (default: 100_000).")
parser.add_argument("--n-chains", type=int, default=20, help="Number of top chains to print/save per weight key (default: 20).")
parser.add_argument("--n-fan-outs", type=int, default=15, help="Number of top fan-out senders to report (default: 15).")
parser.add_argument("--leiden-resolution", type=float, default=5.0, help="Resolution parameter for Leiden community detection (default: 5).")
args = parser.parse_args()


def load_dag(name: str) -> dict:
    path = os.path.join(art_dir, f"dag_{name}.pt")
    print(f"Loading {path} ...")
    return torch.load(path, map_location="cpu")


# ============================================================================
print(f"\n========== Loading DAG ({args.dataset}) ==========")
dag = load_dag(args.dataset)
print(f"  edge weight tensors available: {[k for k in dag if hasattr(dag[k], 'shape') and dag[k].dim() == 4]}")


# ============================================================================
print("\n========== Chain enumeration (per-sender top-K sparsification) ==========")
chains_by_weight = {}
for weight_key in ["APS", "AARV"]:
    if weight_key not in dag:
        print(f"  skipping {weight_key}: not in DAG")
        continue
    print(f"\n--- Chains by {weight_key} ---")
    g = dag_to_igraph(dag, weight_key=weight_key,
                      sparsification="per_sender_topk", K=args.top_K)
    print(f"  sparsified graph: {g.vcount()} nodes, {g.ecount()} edges")

    chains = extract_chains(g,
                            min_length=args.chain_min_length,
                            max_length=args.chain_max_length,
                            weighted_score="min", 
                            max_paths=args.max_paths)
    print(f"  enumerated {len(chains)} paths of length [{args.chain_min_length}, {args.chain_max_length}]")
    print(f"\n  Top {args.n_chains} chains by min-edge-weight:")
    for c in chains[:args.n_chains]:
        ws = [f"{w:+.3f}" for w in c["weights"]]
        print(f"    {chain_str(c):<60}   weights=[{', '.join(ws)}]   min={c['score']:.3f}")
    chains_by_weight[weight_key] = chains[:args.n_chains]

# Inhibitor chains separately (sparsify on |ANS|, sign-flip).
print(f"\n--- Inhibitor chains by |ANS| ---")
g_ans = dag_to_igraph(dag, weight_key="ANS",
                      sparsification="per_sender_topk", K=args.top_K, use_abs=True)
print(f"  sparsified graph: {g_ans.vcount()} nodes, {g_ans.ecount()} edges")
inhibitor_chains = extract_chains(g_ans,
                                   min_length=args.chain_min_length,
                                   max_length=args.chain_max_length,
                                   weighted_score="min",
                                   max_paths=args.max_paths)
print(f"  enumerated {len(inhibitor_chains)} paths")
print(f"\n  Top {args.n_chains} inhibitor chains by min |edge-weight|:")
for c in inhibitor_chains[:args.n_chains]:
    ws = [f"{w:+.3f}" for w in c["weights"]]
    print(f"    {chain_str(c):<60}   weights=[{', '.join(ws)}]   min|w|={c['score']:.3f}")
chains_by_weight["ANS"] = inhibitor_chains[:args.n_chains]


# ============================================================================
print(f"\n========== Community detection (Leiden, γ={args.leiden_resolution}) ==========")
communities_by_weight = {}
for weight_key in ["APS", "AARV"]:
    if weight_key not in dag:
        continue
    print(f"\n--- Communities by {weight_key} ---")
    g = dag_to_igraph(dag, weight_key=weight_key,
                      sparsification="per_sender_topk", K=args.top_K)
    communities = extract_communities(g, resolution=args.leiden_resolution)
    print(f"  found {len(communities)} communities; top-15 by size:")
    summary = []
    for i, members in enumerate(communities[:15]):
        layers = sorted({m // 64 for m in members})
        layer_str = ",".join(str(l) for l in layers[:8])
        if len(layers) > 8:
            layer_str += ",..."
        member_names = [expert_name(m) for m in sorted(members)[:8]]
        more = "" if len(members) <= 8 else f" + {len(members) - 8} more"
        print(f"    #{i+1}: size={len(members):3d}  layers=[{layer_str}]   "
              f"first members: {', '.join(member_names)}{more}")
        summary.append({
            "rank": i + 1,
            "size": len(members),
            "layers": layers,
            "members": [int(m) for m in sorted(members)],
            "member_names": [expert_name(m) for m in sorted(members)],
        })
    communities_by_weight[weight_key] = summary


# ============================================================================
print(f"\n========== Fan-out detection ==========")
fan_outs_by_weight = {}
for weight_key in ["APS", "ANS"]:
    if weight_key not in dag:
        continue
    print(f"\n--- Top fan-outs by {weight_key} ---")
    fan_outs = extract_fan_outs(dag, weight_key=weight_key,
                                 n_targets=8, top_n_senders=args.n_fan_outs,
                                 use_abs=(weight_key == "ANS"))
    for f in fan_outs:
        target_names = ", ".join(f["target_names"])
        print(f"    {f['sender_name']:<8}  conc={f['concentration']:.3f}  "
              f"tot={f['total_outgoing_weight']:+.2f}   targets: {target_names}")
    fan_outs_by_weight[weight_key] = fan_outs


# ============================================================================
results = {
    "dataset": args.dataset,
    "config": {
        "top_K": args.top_K,
        "chain_min_length": args.chain_min_length,
        "chain_max_length": args.chain_max_length,
        "max_paths": args.max_paths, 
        "leiden_resolution": args.leiden_resolution,
    },
    "chains": chains_by_weight,
    "communities": communities_by_weight,
    "fan_outs": fan_outs_by_weight,
}
out_path = os.path.join(art_dir, f"candidates_{args.dataset}.json")
with open(out_path, "w") as f:
    json.dump(results, f, indent=2, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
print(f"\nSaved {out_path}")


# ============================================================================
if args.compare_with is not None:
    other = args.compare_with
    print(f"\n========== Differential analysis ({args.dataset} vs {other}) ==========")
    dag_b = load_dag(other)
    for weight_key in ["AARV", "APS"]:
        if weight_key not in dag or weight_key not in dag_b:
            continue
        print(f"\n--- Top differential edges by {weight_key} ---")
        diffs = differential_edges(dag, dag_b, weight_key=weight_key,
                                    n_edges=20, min_max_weight=0.1)
        for d in diffs:
            print(f"    {d['sender_name']:<8} → {d['receiver_name']:<8}   "
                  f"w_{args.dataset}={d['w_a']:+7.3f}  w_{other}={d['w_b']:+7.3f}   "
                  f"ratio={d['ratio']:.3f}")

    diff_path = os.path.join(art_dir, f"differential_{args.dataset}_vs_{other}.json")
    diff_results = {}
    for weight_key in ["AARV", "APS"]:
        if weight_key not in dag or weight_key not in dag_b:
            continue
        diff_results[weight_key] = differential_edges(dag, dag_b, weight_key=weight_key,
                                                      n_edges=50, min_max_weight=0.1)
    with open(diff_path, "w") as f:
        json.dump({
            "datasets": [args.dataset, other],
            "differential_edges": diff_results,
        }, f, indent=2)
    print(f"\nSaved {diff_path}")
