"""Matched top-K overlap between act(v) (Su et al.) and out(v) (ours).

Fair comparison of the two importance statistics, decoupled from any
threshold choice: rank all experts by act(v) and by out(v) over the SAME
layer-filtered population (Su's include_layers filter applied to both),
take the top-K of each, and measure the overlap.

For each model:
  - overlap@K = |topK_act ∩ topK_out| for K in K_GRID
  - chance level E[|∩|] = K^2 / V for random size-K sets
  - side-by-side top-K lists at --k-show, intersection marked with '*'

Layer labels use Su's model-absolute convention (DeepSeek: DAG layer + 1).

Usage:
    python experiments/overlap_topk.py
    python experiments/overlap_topk.py --include-layers 1.0   # full network
    python experiments/overlap_topk.py --k-show 10 --task c4
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.cross_rank_analysis import (  # noqa: E402
    MODELS, NUM_DENSE, _load,
)

K_GRID = [1, 2, 3, 5, 10, 20]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--include-layers", type=float, default=0.75,
                   help="Layer-depth cutoff applied to BOTH rankings "
                        "(default 0.75 = Su's setting; 1.0 = full network).")
    p.add_argument("--k-show", type=int, default=10,
                   help="K for the side-by-side expert lists (default 10).")
    args = p.parse_args()

    print(f"Matched top-K overlap of act(v) vs out(v)  "
          f"(task={args.task}, include_layers={args.include_layers})")
    header = (f"{'model':<18s} {'V_filt':>7s}  "
              + "  ".join(f"{'ov@' + str(k):>6s}" for k in K_GRID)
              + f"  {'chance@10':>9s}")
    print(header)
    print("-" * len(header))

    details = {}
    for m in MODELS:
        d = _load(m, args.task)
        if d is None:
            print(f"{m:<18s}  MISSING")
            continue
        L, N = d["L"], d["N"]
        nd = NUM_DENSE.get(m, 0)
        # Su's layer filter on model-absolute indices, applied to BOTH
        # statistics so the ranked populations are identical.
        upto = max(0, min(L, round((L + nd) * args.include_layers) - nd))
        keep = (d["depth"] < upto)
        idx = np.flatnonzero(keep)
        act_f, out_f = d["act"][keep], d["out"][keep]
        V = idx.size

        order_act = idx[np.argsort(-act_f)]
        order_out = idx[np.argsort(-out_f)]

        ovs = []
        for k in K_GRID:
            kk = min(k, V)
            ovs.append(len(set(order_act[:kk].tolist())
                           & set(order_out[:kk].tolist())))
        chance10 = (min(10, V) ** 2) / V
        print(f"{m:<18s} {V:>7d}  "
              + "  ".join(f"{o:>6d}" for o in ovs)
              + f"  {chance10:>9.3f}")
        details[m] = (order_act, order_out, d, nd, N)

    # ---- Side-by-side lists at K = k_show -------------------------------
    K = args.k_show
    for m, (order_act, order_out, d, nd, N) in details.items():
        inter = set(order_act[:K].tolist()) & set(order_out[:K].tolist())

        def lab(flat: int) -> str:
            return f"L{flat // N + nd}E{flat % N}"

        print(f"\n[{m}]  top-{K} by act vs top-{K} by out   "
              f"(* = in both, |∩| = {len(inter)})")
        print(f"  {'rank':>4s}  {'by act':>12s} {'act(v)':>10s}   "
              f"{'by out':>12s} {'out(v)':>10s}")
        for r in range(min(K, len(order_act))):
            fa, fo = int(order_act[r]), int(order_out[r])
            ma = "*" if fa in inter else " "
            mo = "*" if fo in inter else " "
            print(f"  {r + 1:>4d}  {lab(fa) + ma:>12s} {d['act'][fa]:>10.4g}   "
                  f"{lab(fo) + mo:>12s} {d['out'][fo]:>10.4g}")


if __name__ == "__main__":
    main()
