"""Sanity-check the random-init pilot DAGs.

For each pilot DAG (mixtral-8x7b/c4, seeds 0, 1, 2), we report:
  1. Per-layer load Gini coefficient -- catches degenerate routing
     (a single expert receiving all tokens has Gini -> 1).
  2. Per-layer routing entropy in bits, normalised by log2(N_experts).
     Near-uniform routing -> 1; collapsed routing -> 0.
  3. Mean activation magnitude -- catches NaN / inf / pathological scales.
  4. Edge-weight (W_softmax) distribution summary.

Pass thresholds (sanity-only, not statistical):
  - All layers: load Gini <= 0.5  (well below 1 = winner-take-all)
  - All layers: normalised routing entropy >= 0.7
  - act values: all finite, max < 1e4

Run on the cluster after the pilot array finishes:
    python experiments/check_random_init_pilot.py

Reads: ${result_path}/circuits/dag_mixtral-8x7b_c4_rand_s{0,1,2}.pt
Writes nothing; prints a pass/fail report.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DATASET = "c4"
DEFAULT_SEEDS = [0, 1, 2]
GINI_MAX = 0.5
ENTROPY_NORM_MIN = 0.7
ACT_MAX = 1e4


def gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative 1D array. Returns 0 for uniform, ~1
    for fully concentrated. NaN for all-zero input."""
    x = np.asarray(x, dtype=np.float64)
    if x.sum() <= 0:
        return float("nan")
    x_sorted = np.sort(x)
    n = len(x_sorted)
    cum = np.cumsum(x_sorted)
    return float((n + 1 - 2 * cum.sum() / cum[-1]) / n)


def entropy_bits(probs: np.ndarray) -> float:
    """Shannon entropy of a probability vector in bits."""
    p = probs[probs > 0]
    if p.size == 0:
        return 0.0
    return float(-np.sum(p * np.log2(p)))


def check_one(path: Path) -> tuple[bool, str]:
    if not path.exists():
        return False, f"MISSING: {path}"
    try:
        dag = torch.load(path, weights_only=False)
    except Exception as e:
        return False, f"LOAD ERROR: {e}"

    n_tok = dag["n_tokens_selected"].cpu().numpy().astype(np.float64)  # [L, N]
    act = dag["act"].cpu().numpy().astype(np.float64)                 # [L, N]
    W = dag["W_softmax"].cpu().numpy().astype(np.float64)             # [L, N, L, N]
    L, N = n_tok.shape
    log2N = float(np.log2(N))

    # 1) Per-layer load Gini.
    ginis = np.array([gini(n_tok[ell]) for ell in range(L)])

    # 2) Per-layer routing entropy normalised by log2(N).
    layer_probs = n_tok / np.maximum(n_tok.sum(axis=1, keepdims=True), 1.0)
    ent = np.array([entropy_bits(layer_probs[ell]) for ell in range(L)]) / log2N

    # 3) Activation summary.
    act_finite = np.isfinite(act).all()
    act_max = float(np.nanmax(act))

    # 4) Edge weights.
    W_pos = W[W > 0]
    W_mean = float(W_pos.mean()) if W_pos.size > 0 else float("nan")
    W_max = float(W_pos.max()) if W_pos.size > 0 else float("nan")

    print(f"\n{path.name}")
    print(f"  L = {L}, N = {N}")
    print(f"  load Gini per layer    : min={ginis.min():.3f}  median={float(np.median(ginis)):.3f}  max={ginis.max():.3f}")
    print(f"  routing entropy (norm) : min={ent.min():.3f}  median={float(np.median(ent)):.3f}  max={ent.max():.3f}")
    print(f"  act values             : finite={act_finite}  max={act_max:.3e}")
    print(f"  W_softmax (positive)   : count={W_pos.size}  mean={W_mean:.3e}  max={W_max:.3e}")

    fails = []
    if (ginis > GINI_MAX).any():
        bad = int((ginis > GINI_MAX).sum())
        fails.append(f"{bad}/{L} layers with load Gini > {GINI_MAX}")
    if (ent < ENTROPY_NORM_MIN).any():
        bad = int((ent < ENTROPY_NORM_MIN).sum())
        fails.append(f"{bad}/{L} layers with normalised entropy < {ENTROPY_NORM_MIN}")
    if not act_finite:
        fails.append("act contains non-finite values")
    if act_max > ACT_MAX:
        fails.append(f"act_max = {act_max:.1e} > {ACT_MAX:.0e}")

    if fails:
        return False, "FAIL: " + "; ".join(fails)
    return True, "PASS"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="mixtral-8x7b",
                        help="Model name to sanity-check (default: mixtral-8x7b).")
    parser.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS,
                        help="Random-init seeds to check (default: 0 1 2).")
    parser.add_argument("--dataset", default=DATASET,
                        help=f"Dataset suffix in the DAG filename (default: {DATASET}).")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    circuits_dir = Path(cfg["result_path"]) / "circuits"

    print(f"Sanity-check: {args.model} random-init DAGs on {args.dataset}")
    print("=" * 70)

    results = []
    for seed in args.seeds:
        p = circuits_dir / f"dag_{args.model}_{args.dataset}_rand_s{seed}.pt"
        ok, msg = check_one(p)
        results.append((seed, ok, msg))

    print("\n" + "=" * 70)
    print("Summary:")
    for seed, ok, msg in results:
        marker = "OK   " if ok else "FAIL "
        print(f"  seed={seed}  {marker}  {msg}")

    all_ok = all(ok for _, ok, _ in results)
    if all_ok:
        print(f"\nAll seeds passed sanity checks for {args.model}.")
        sys.exit(0)
    else:
        print(f"\nAt least one seed failed for {args.model}. Investigate before proceeding.")
        sys.exit(1)


if __name__ == "__main__":
    main()
