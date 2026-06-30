"""Diagnose why headline and LOO sweeps disagree at alpha=0.

Picks one model pair + one task + one Q, then:
  1. Builds the triple via the LOO pipeline (_build_filtered_triple).
  2. Builds the triple via the headline pipeline (build_triple_at_Q).
  3. Computes fgw_distance(alpha=0) for each.
  4. Reads the corresponding cell from each npz.
  5. Prints all four values side-by-side.

If FRESH(LOO) == FRESH(headline) but NPZ values differ -> one or both npzs are stale.
If FRESH(LOO) != FRESH(headline)                       -> the pipelines actually differ.

Run on piora:
    python3 experiments/diagnose_alpha0_drift.py \
        --model-i deepseek-v2-lite --model-j deepseek-v2 \
        --task c4 --Q 0.9
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import yaml
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.fgw import fgw_distance
from experiments.feature_ablation_sweep import (
    _build_filtered_triple, q_threshold,
)
from experiments.run_alpha_beta_sweep import (
    build_triple_at_Q, MODELS, TASKS, QUANTILES,
    load_dag, load_classification,
)

ALPHAS = [0.0, 0.5, 1.0]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model-i", required=True)
    p.add_argument("--model-j", required=True)
    p.add_argument("--task", default="c4")
    p.add_argument("--Q", type=float, default=0.9)
    p.add_argument("--n-init", type=int, default=3)
    p.add_argument("--act-norm", default="log_max", choices=["rank", "log_max"])
    p.add_argument("--load-norm", default="log_max", choices=["raw", "log_max"])
    args = p.parse_args()

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]

    # --- 1. LOO pipeline triple ---
    print(f"\n[1/4] LOO pipeline build_triple ...")
    dag_i = load_dag(args.model_i, args.task)
    dag_j = load_dag(args.model_j, args.task)
    cls_i = load_classification(args.model_i, args.task)
    cls_j = load_classification(args.model_j, args.task)
    theta_i = q_threshold(dag_i, args.Q)
    theta_j = q_threshold(dag_j, args.Q)
    t_i_loo, _ = _build_filtered_triple(dag_i, cls_i, theta_i,
                                        act_norm_method=args.act_norm,
                                        load_norm_method=args.load_norm)
    t_j_loo, _ = _build_filtered_triple(dag_j, cls_j, theta_j,
                                        act_norm_method=args.act_norm,
                                        load_norm_method=args.load_norm)

    # --- 2. Headline pipeline triple (structural-mode=conn, gamma=0.5) ---
    print(f"[2/4] Headline pipeline build_triple_at_Q (conn, gamma=0.5) ...")
    t_i_hl, _ = build_triple_at_Q(args.model_i, args.task, cls_i, args.Q,
                                  act_norm_method=args.act_norm,
                                  load_norm_method=args.load_norm,
                                  structural_mode="conn", gamma=0.5)
    t_j_hl, _ = build_triple_at_Q(args.model_j, args.task, cls_j, args.Q,
                                  act_norm_method=args.act_norm,
                                  load_norm_method=args.load_norm,
                                  structural_mode="conn", gamma=0.5)

    # --- 3. Are F, mass, vertex-set identical between LOO and headline? ---
    print(f"\n[3/4] Static comparison of triples ...")
    for who, t_loo, t_hl in [("i", t_i_loo, t_i_hl), ("j", t_j_loo, t_j_hl)]:
        C_l, F_l, m_l, _ = t_loo
        C_h, F_h, m_h, _ = t_hl
        print(f"  model {who}: "
              f"|V| loo={len(m_l)} headline={len(m_h)}  "
              f"F match={np.allclose(F_l, F_h)}  "
              f"mass match={np.allclose(m_l, m_h)}  "
              f"C match={np.allclose(C_l, C_h)}  "
              f"max|F1-F2|={np.abs(F_l - F_h).max() if F_l.shape == F_h.shape else 'shape-mismatch'}")

    # --- 4. FGW at alpha=0 via each triple. Should be identical if F/mass agree. ---
    print(f"\n[4/4] fgw_distance(alpha=0, n_init={args.n_init}) ...")
    for alpha in ALPHAS:
        d_loo, _ = fgw_distance(t_i_loo, t_j_loo, alpha=alpha, n_init=args.n_init, seed=0)
        d_hl,  _ = fgw_distance(t_i_hl,  t_j_hl,  alpha=alpha, n_init=args.n_init, seed=0)
        s_loo = float(np.exp(-d_loo))
        s_hl  = float(np.exp(-d_hl))
        print(f"  alpha={alpha}:  S_loo={s_loo:.6f}  S_hl={s_hl:.6f}  delta={s_hl - s_loo:+.6f}")

    # --- 5. What does each npz report for the same cell? ---
    print(f"\nWhat each npz currently reports for ({args.model_i} x {args.model_j}, {args.task}, Q={args.Q}, alpha=0):")
    # Headline npz
    suffix_h = f"_logact_logload_conn_g0.5" if (args.act_norm == "log_max" and args.load_norm == "log_max") else ""
    hl_npz = Path(result_path) / "circuits" / f"alpha_beta_sweep{suffix_h}" / "S_full_with_act.npz"
    if hl_npz.exists():
        H = np.load(hl_npz, allow_pickle=True)
        HS = H['S']
        TUPLES = [(m, t) for m in MODELS for t in TASKS]
        ti = TUPLES.index((args.model_i, args.task)); tj = TUPLES.index((args.model_j, args.task))
        qi = int(np.argmin(np.abs(np.asarray(H['quantiles']) - args.Q)))
        s_hl_npz = float(HS[ti, tj, 0, qi])  # alpha_idx=0 -> alpha=0
        print(f"  headline npz S = {s_hl_npz:.6f}   path={hl_npz}")
    # LOO npz
    suffix_l = "_logact_logload" if (args.act_norm == "log_max" and args.load_norm == "log_max") else ""
    loo_npz = Path(result_path) / "circuits" / f"feature_ablation{suffix_l}" / "S_loo.npz"
    if loo_npz.exists():
        L = np.load(loo_npz, allow_pickle=True)
        LS = L['S']  # (abl, Q, task, M1, M2)
        ablations = list(L['ablations'])
        full_idx = ablations.index('full')
        qi = int(np.argmin(np.abs(np.asarray(L['quantiles']) - args.Q)))
        mi = MODELS.index(args.model_i); mj = MODELS.index(args.model_j)
        ti = TASKS.index(args.task)
        s_loo_npz = float(LS[full_idx, qi, ti, mi, mj])
        print(f"  loo npz S      = {s_loo_npz:.6f}   path={loo_npz}")


if __name__ == "__main__":
    main()
