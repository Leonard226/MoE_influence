"""Matched top-K overlap between act(v) (Su et al.) and out(v) (ours).

Fair comparison of the two importance statistics, decoupled from any
threshold choice: rank experts by each statistic AS ITS METHOD PRESCRIBES,
take the top-K of each, and measure the overlap.

  - act(v) is ranked over Su et al.'s layer-filtered population
    (model-absolute layer < round(include_layers * L)); the filter is an
    integral part of their method (without it the ranking floods with
    late-layer massive-activation receivers).
  - out(v) is ranked over the FULL network: it requires no depth filter
    (sink receivers rank low automatically), though note it is also
    structurally unable to rank late-layer experts high (out -> 0 at the
    last layer by construction).

Chance level for random size-K sets (act from V_filt, out from V_full):
E[|A ∩ B|] = K^2 / V_full.

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
                   help="Layer-depth cutoff applied to the act(v) ranking "
                        "ONLY (default 0.75 = Su's setting; 1.0 = no filter). "
                        "out(v) is always ranked over the full network.")
    p.add_argument("--k-show", type=int, default=10,
                   help="K for the side-by-side expert lists (default 10).")
    args = p.parse_args()

    print(f"Matched top-K overlap of act(v) vs out(v)  "
          f"(task={args.task}, include_layers={args.include_layers} "
          f"on act only; out unfiltered)")
    header = (f"{'model':<18s} {'V_filt':>7s} {'V_full':>7s}  "
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
        # Su's layer filter (model-absolute indices) applies to the act
        # ranking only -- it is part of their method. out is ranked over
        # the full network.
        upto = max(0, min(L, round((L + nd) * args.include_layers) - nd))
        idx_filt = np.flatnonzero(d["depth"] < upto)
        V_filt, V_full = idx_filt.size, d["out"].size

        order_act = idx_filt[np.argsort(-d["act"][idx_filt])]
        order_out = np.argsort(-d["out"])

        ovs = []
        for k in K_GRID:
            ovs.append(len(set(order_act[:min(k, V_filt)].tolist())
                           & set(order_out[:min(k, V_full)].tolist())))
        chance10 = (min(10, V_filt) * min(10, V_full)) / V_full
        print(f"{m:<18s} {V_filt:>7d} {V_full:>7d}  "
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
