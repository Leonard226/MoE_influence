"""Compare rank-norm vs log-max-norm ablation results on the c4 task only.

This is the validation script for the act-normalisation change: it loads
the per-pair-task files from the two output directories and aggregates
c4-only within/cross-family means under both normalisations, printed
side by side.

Reads (rank, legacy):
  results/circuits/feature_ablation/S_loo_pair_NNN.npz       for NNN in {0, 8, 16, ..., 216}
Reads (log_max, new):
  results/circuits/feature_ablation_logact/S_loo_pair_NNN.npz for the same NNN

(NNN = pair_idx * 8 + 0; t_idx = 0 corresponds to c4.)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


MODELS = [
    "mixtral-8x7b", "mixtral-8x22b",
    "deepseek-v2-lite", "deepseek-v2",
    "qwen3-30b-a3b", "qwen3-235b-a22b",
    "olmoe", "phi-3.5-moe",
]
ABLATIONS = ["full", "no_depth", "no_out", "no_in", "no_load", "no_act", "no_class"]
QUANTILES = [0.9, 0.99, 0.999]
FAMILIES = {
    "Mixtral":  [0, 1],
    "DeepSeek": [2, 3],
    "Qwen3":    [4, 5],
}
MODEL_TO_FAMILY = {m: f for f, idxs in FAMILIES.items() for m in idxs}
N_T = 8         # tasks per pair_task_idx encoding (see feature_ablation_sweep.py)
T_IDX_C4 = 0    # c4 is the first task in TASKS in feature_ablation_sweep.py


def load_c4_pairs(dir_path: Path) -> dict[tuple[int, int], np.ndarray]:
    """Load c4 pair-task results from `dir_path`. Returns
    {(mi, mj) -> ndarray shape (n_abl, n_q)}."""
    results: dict[tuple[int, int], np.ndarray] = {}
    n_pairs = len(MODELS) * (len(MODELS) - 1) // 2
    for pair_idx in range(n_pairs):
        pair_task_idx = pair_idx * N_T + T_IDX_C4
        npz_path = dir_path / f"S_loo_pair_{pair_task_idx:03d}.npz"
        if not npz_path.exists():
            continue
        d = np.load(npz_path, allow_pickle=True)
        mi = int(d["mi"])
        mj = int(d["mj"])
        results[(mi, mj)] = np.asarray(d["S"])
    return results


def aggregate(results: dict[tuple[int, int], np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    n_abl, n_q = len(ABLATIONS), len(QUANTILES)
    within = np.full((n_abl, n_q), np.nan, dtype=np.float64)
    cross  = np.full((n_abl, n_q), np.nan, dtype=np.float64)
    for abl_idx in range(n_abl):
        for q_idx in range(n_q):
            w_vals: list[float] = []
            c_vals: list[float] = []
            for (mi, mj), S in results.items():
                s = float(S[abl_idx, q_idx])
                if np.isnan(s):
                    continue
                fam_i = MODEL_TO_FAMILY.get(mi)
                fam_j = MODEL_TO_FAMILY.get(mj)
                same_fam = (fam_i is not None and fam_i == fam_j)
                (w_vals if same_fam else c_vals).append(s)
            if w_vals:
                within[abl_idx, q_idx] = float(np.mean(w_vals))
            if c_vals:
                cross[abl_idx, q_idx] = float(np.mean(c_vals))
    return within, cross


def main() -> None:
    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    rank_dir   = Path(result_path) / "circuits" / "feature_ablation"
    logact_dir = Path(result_path) / "circuits" / "feature_ablation_logact"

    rank_res   = load_c4_pairs(rank_dir)
    logact_res = load_c4_pairs(logact_dir)
    n_pairs = len(MODELS) * (len(MODELS) - 1) // 2

    print(f"rank   (legacy) pairs loaded: {len(rank_res):>3d} / {n_pairs}   from {rank_dir.name}/")
    print(f"logact (new)    pairs loaded: {len(logact_res):>3d} / {n_pairs}   from {logact_dir.name}/")
    if not rank_res or not logact_res:
        print("Missing data in one of the directories; cannot compare.")
        sys.exit(1)
    print()

    rank_w, rank_c     = aggregate(rank_res)
    logact_w, logact_c = aggregate(logact_res)

    print("c4-only within/cross-family means per (ablation, Q):")
    print(f"{'Ablation':<10s}  {'Q':>6s}  "
          f"{'w(rank)':>9s} {'w(log)':>9s} {'Δw':>8s}    "
          f"{'c(rank)':>9s} {'c(log)':>9s} {'Δc':>8s}")
    print("-" * 92)
    for abl_idx, abl in enumerate(ABLATIONS):
        for q_idx, Q in enumerate(QUANTILES):
            rw, lw = rank_w[abl_idx, q_idx], logact_w[abl_idx, q_idx]
            rc, lc = rank_c[abl_idx, q_idx], logact_c[abl_idx, q_idx]
            print(f"{abl:<10s}  {Q:>6.3f}  "
                  f"{rw:>9.4f} {lw:>9.4f} {lw - rw:>+8.4f}    "
                  f"{rc:>9.4f} {lc:>9.4f} {lc - rc:>+8.4f}")
        print()

    # Key question: did the -act row's effect change materially?
    print("=" * 92)
    print("Headline check: did the -act effect change under log_max?")
    print("=" * 92)
    full_idx = ABLATIONS.index("full")
    act_idx  = ABLATIONS.index("no_act")
    load_idx = ABLATIONS.index("no_load")
    print(f"{'metric':<35s}  {'Q':>6s}  {'rank':>9s}  {'log_max':>9s}")
    print("-" * 70)
    for q_idx, Q in enumerate(QUANTILES):
        # Within delta: -act lifts within mean by how much?
        dw_act_rank = rank_w[act_idx, q_idx]   - rank_w[full_idx, q_idx]
        dw_act_log  = logact_w[act_idx, q_idx] - logact_w[full_idx, q_idx]
        # Cross delta: -act lifts cross mean by how much?
        dc_act_rank = rank_c[act_idx, q_idx]   - rank_c[full_idx, q_idx]
        dc_act_log  = logact_c[act_idx, q_idx] - logact_c[full_idx, q_idx]
        # Compare to -load for context.
        dc_load_rank = rank_c[load_idx, q_idx]   - rank_c[full_idx, q_idx]
        dc_load_log  = logact_c[load_idx, q_idx] - logact_c[full_idx, q_idx]
        print(f"{'-act Δw (within-fam shift)':<35s}  {Q:>6.3f}  "
              f"{dw_act_rank:>+9.4f}  {dw_act_log:>+9.4f}")
        print(f"{'-act Δc (cross-fam shift)':<35s}  {Q:>6.3f}  "
              f"{dc_act_rank:>+9.4f}  {dc_act_log:>+9.4f}")
        print(f"{'-load Δc (for context)':<35s}  {Q:>6.3f}  "
              f"{dc_load_rank:>+9.4f}  {dc_load_log:>+9.4f}")
        print()


if __name__ == "__main__":
    main()
