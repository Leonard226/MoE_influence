"""Compare the spectral baseline against the FGW headline sweep.

For each (alpha, Q) cell, computes:
  rho_spearman[alpha, Q] = Spearman rho between S_spec[:, :, Q]
                                              and S_fgw[:, :, alpha, Q]
  rho_pearson[alpha, Q]  = Pearson rho on the same pairs.

Correlations are computed on the off-diagonal UPPER-TRIANGLE pairs only
(diagonal is trivially 1; lower triangle is the symmetric mirror).

Also reports per-Q descriptive statistics of S_spec, and a per-(alpha, Q)
within/cross-family decomposition (to check whether the spectral baseline
preserves the within/cross gap that FGW shows).

Inputs:
  ${result_path}/circuits/spectral_baseline/S_struct.npz
  ${result_path}/circuits/alpha_beta_sweep_logact_logload/S_full_with_act.npz

Output:
  ${result_path}/circuits/spectral_baseline/comparison_summary.txt
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import scipy.stats
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from experiments.run_alpha_beta_sweep import (   # noqa: E402
    MODELS, TASKS, TUPLES, N_TUPLES, ALPHAS, QUANTILES,
)

with open(os.path.join(ROOT, "config.yaml")) as f:
    _config = yaml.safe_load(f)
CIRCUITS_DIR = Path(_config["result_path"]) / "circuits"

SPEC_PATH = CIRCUITS_DIR / "spectral_baseline" / "S_struct.npz"
FGW_PATH = CIRCUITS_DIR / "alpha_beta_sweep_logact_logload" / "S_full_with_act.npz"
OUT_PATH = CIRCUITS_DIR / "spectral_baseline" / "comparison_summary.txt"

FAMILIES = {
    "Mixtral":  ["mixtral-8x7b", "mixtral-8x22b"],
    "DeepSeek": ["deepseek-v2-lite", "deepseek-v2"],
    "Qwen3":    ["qwen3-30b-a3b", "qwen3-235b-a22b"],
}
MODEL_TO_FAMILY = {m: f for f, ms in FAMILIES.items() for m in ms}


def _upper_tri_pairs(S: np.ndarray) -> np.ndarray:
    """Return upper-triangular off-diagonal values of an (N, N) matrix."""
    iu, ju = np.triu_indices(S.shape[0], k=1)
    return S[iu, ju]


def _tuple_to_family(tup_idx: int) -> str | None:
    model, _ = TUPLES[tup_idx]
    return MODEL_TO_FAMILY.get(model)


def _same_task_within_cross(S: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (within_family, cross_family) arrays of same-task off-diagonal S
    values. Mirrors collect_same_task in analyze_alpha_beta_sweep.py."""
    within, cross = [], []
    for t in TASKS:
        for mi, m1 in enumerate(MODELS):
            for mj, m2 in enumerate(MODELS):
                if mj <= mi:
                    continue
                i = TUPLES.index((m1, t))
                j = TUPLES.index((m2, t))
                v = S[i, j]
                if np.isnan(v):
                    continue
                fam1 = MODEL_TO_FAMILY.get(m1)
                fam2 = MODEL_TO_FAMILY.get(m2)
                same_fam = (fam1 is not None and fam1 == fam2)
                (within if same_fam else cross).append(v)
    return np.array(within), np.array(cross)


def main() -> None:
    if not SPEC_PATH.exists():
        print(f"ERROR: missing spectral tensor {SPEC_PATH}")
        print("  Run experiments/run_spectral_baseline.py first.")
        sys.exit(1)
    if not FGW_PATH.exists():
        print(f"ERROR: missing FGW tensor {FGW_PATH}")
        sys.exit(1)

    spec = np.load(SPEC_PATH, allow_pickle=True)
    fgw = np.load(FGW_PATH, allow_pickle=True)
    S_spec = spec["S"]                                  # (N, N, n_q)
    S_fgw = fgw["S"].astype(np.float64)                 # (N, N, n_alpha, n_q)
    v_effs = spec["v_effs"]                             # (N, n_q)

    assert S_spec.shape == (N_TUPLES, N_TUPLES, len(QUANTILES)), \
        f"S_spec shape {S_spec.shape} != expected"
    assert S_fgw.shape == (N_TUPLES, N_TUPLES, len(ALPHAS), len(QUANTILES)), \
        f"S_fgw shape {S_fgw.shape} != expected"

    lines: list[str] = []
    def w(s: str = "") -> None:
        print(s)
        lines.append(s)

    w("=" * 78)
    w("Spectral (magnetic-Laplacian NetLSD) vs FGW comparison")
    w("=" * 78)
    w(f"FGW tensor    : {FGW_PATH}")
    w(f"Spectral tens.: {SPEC_PATH}")
    w(f"  q_phase     : {float(spec['q_phase'])}")
    w(f"  edge_tensor : {str(spec['edge_tensor'])}")
    w(f"  n_t (NetLSD): {S_spec.shape[2] if S_spec.ndim < 4 else spec['signatures'].shape[2]}")
    w("")

    # --- Effective vertex count per (model, Q) ---------------------------
    w("Effective vertex count after Q-sparsification + isolated-vertex filter:")
    w(f"  {'model':<18s}  {'task':<11s}  " +
      "  ".join(f"Q={Q:5.3g}" for Q in QUANTILES))
    w("  " + "-" * 70)
    for ti, (model, task) in enumerate(TUPLES[:8]):  # first 8 rows for brevity
        w(f"  {model:<18s}  {task:<11s}  " +
          "  ".join(f"{v_effs[ti, qi]:>7d}" for qi in range(len(QUANTILES))))
    w(f"  ... ({N_TUPLES - 8} more rows, one per (model, task))")
    w("")

    # --- Correlation table -----------------------------------------------
    w("=" * 78)
    w("Spearman / Pearson correlation: S_spec vs FGW S_alpha")
    w("=" * 78)
    w("  Upper-triangular off-diagonal pairs (n = N*(N-1)/2 = "
      f"{N_TUPLES*(N_TUPLES-1)//2}). NaN pairs excluded.\n")

    n_alpha, n_q = len(ALPHAS), len(QUANTILES)
    rho_sp = np.zeros((n_alpha, n_q))
    rho_pe = np.zeros((n_alpha, n_q))

    for ai, alpha in enumerate(ALPHAS):
        for qi, Q in enumerate(QUANTILES):
            s_spec = _upper_tri_pairs(S_spec[:, :, qi])
            s_fgw  = _upper_tri_pairs(S_fgw[:, :, ai, qi])
            valid = ~(np.isnan(s_spec) | np.isnan(s_fgw))
            rho_sp[ai, qi], _ = scipy.stats.spearmanr(s_spec[valid], s_fgw[valid])
            rho_pe[ai, qi], _ = scipy.stats.pearsonr(s_spec[valid], s_fgw[valid])

    w(f"  Spearman:  {'alpha\\Q':<10s}  " + "  ".join(f"Q={Q:5.3g}" for Q in QUANTILES))
    w("  " + "-" * 60)
    for ai, alpha in enumerate(ALPHAS):
        w(f"             alpha = {alpha:.2f}  " +
          "  ".join(f"{rho_sp[ai, qi]:+7.3f}" for qi in range(n_q)))
    w("")
    w(f"  Pearson:   {'alpha\\Q':<10s}  " + "  ".join(f"Q={Q:5.3g}" for Q in QUANTILES))
    w("  " + "-" * 60)
    for ai, alpha in enumerate(ALPHAS):
        w(f"             alpha = {alpha:.2f}  " +
          "  ".join(f"{rho_pe[ai, qi]:+7.3f}" for qi in range(n_q)))
    w("")

    # --- Within/cross-family same-task decomposition ---------------------
    w("=" * 78)
    w("Within-family vs cross-family same-task means")
    w("=" * 78)
    w("  Spectral baseline:")
    w(f"  {'Q':<6s}  {'within (n=24)':>14s}  {'cross (n=200)':>14s}  "
      f"{'gap':>8s}  {'ratio':>7s}")
    w("  " + "-" * 60)
    for qi, Q in enumerate(QUANTILES):
        within, cross = _same_task_within_cross(S_spec[:, :, qi])
        if len(within) and len(cross):
            gap = within.mean() - cross.mean()
            ratio = within.mean() / cross.mean()
            w(f"  {Q:<6.3g}  {within.mean():>14.4f}  {cross.mean():>14.4f}  "
              f"{gap:>+8.4f}  {ratio:>6.2f}x")
    w("")
    w("  For reference -- FGW at alpha = 0 (pure feature, the discriminative regime):")
    w(f"  {'Q':<6s}  {'within (n=24)':>14s}  {'cross (n=200)':>14s}  "
      f"{'gap':>8s}  {'ratio':>7s}")
    w("  " + "-" * 60)
    for qi, Q in enumerate(QUANTILES):
        within, cross = _same_task_within_cross(S_fgw[:, :, 0, qi])
        if len(within) and len(cross):
            gap = within.mean() - cross.mean()
            ratio = within.mean() / cross.mean()
            w(f"  {Q:<6.3g}  {within.mean():>14.4f}  {cross.mean():>14.4f}  "
              f"{gap:>+8.4f}  {ratio:>6.2f}x")
    w("")
    w("  For reference -- FGW at alpha = 1 (pure structure):")
    w(f"  {'Q':<6s}  {'within (n=24)':>14s}  {'cross (n=200)':>14s}  "
      f"{'gap':>8s}  {'ratio':>7s}")
    w("  " + "-" * 60)
    for qi, Q in enumerate(QUANTILES):
        within, cross = _same_task_within_cross(S_fgw[:, :, 2, qi])
        if len(within) and len(cross):
            gap = within.mean() - cross.mean()
            ratio = within.mean() / cross.mean()
            w(f"  {Q:<6.3g}  {within.mean():>14.4f}  {cross.mean():>14.4f}  "
              f"{gap:>+8.4f}  {ratio:>6.2f}x")
    w("")

    # --- Read-off ---------------------------------------------------------
    w("=" * 78)
    w("Reading the correlation table")
    w("=" * 78)
    w("  High at alpha=1, drops at alpha=0:")
    w("    Spectral tracks FGW's structural channel; FGW's feature channel is")
    w("    genuinely different. Validates FGW's alpha=1 as a structural metric.")
    w("  High across all alpha:")
    w("    Routing similarity is dominantly structural; features are largely")
    w("    redundant with what the magnetic-Laplacian spectrum already captures.")
    w("  Low at alpha=1 too:")
    w("    Magnetic-Laplacian spectrum does NOT capture what FGW's C_path does.")
    w("    Both metrics may be saturating on a structurally homogeneous corpus.")
    w("")

    OUT_PATH.write_text("\n".join(lines))
    print(f"\nSaved: {OUT_PATH}")


if __name__ == "__main__":
    main()
