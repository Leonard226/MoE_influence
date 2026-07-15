"""Cross-rank analysis of Influential (top-out) vs Super (act) vs Sensitive
(top-in) experts.

Step 1 -- kill the cutoff artifact:
  For every expert in the union of
    I = top-K by out(v)          ("Influential Experts", ours)
    S = Su-exact Super Experts   (act(v) criterion, include_layers=0.75)
    N = top-K by in(v)           ("Sensitive Experts", ours)
  report its PERCENTILE in all three distributions (act, out, in), so
  set membership can be judged on the full continuum instead of a
  top-K / threshold cutoff.

Step 2 -- measure the latent "alignment" residual:
  Per model, fit OLS:   log10 out(v) ~ a * log10 act(v) + b * depth + c
  on all experts with act > 0 and out > 0. The residual of an expert is
  the part of its routing influence NOT explained by activation
  magnitude and depth. Prediction from the router-alignment hypothesis:
    SE-only experts        -> large negative residuals
    Influential-only       -> large positive residuals
    Intersection (I and S) -> near zero / positive

Percentile convention: percentage of experts in the same graph with a
strictly smaller value (100 = the top expert).

Layer labels are printed in Su et al.'s model-absolute convention
(DeepSeek models: DAG layer + 1); the raw DAG layer is also shown.

Usage:
    python experiments/cross_rank_analysis.py
    python experiments/cross_rank_analysis.py --task math --top-k 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import yaml


ROOT = Path(__file__).resolve().parent.parent
with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
CIRCUITS = Path(CFG["result_path"]) / "circuits"

MODELS = [
    "mixtral-8x7b", "mixtral-8x22b", "phi-3.5-moe",
    "deepseek-v2-lite", "olmoe",
    "qwen3-30b-a3b", "qwen3-235b-a22b", "deepseek-v2",
]

# Model-absolute layer indexing (Su's convention): our DAG omits the
# first dense layer(s) of the DeepSeek family.
NUM_DENSE = {
    "mixtral-8x7b": 0, "mixtral-8x22b": 0,
    "phi-3.5-moe": 0,
    "deepseek-v2-lite": 1, "deepseek-v2": 1,
    "olmoe": 0,
    "qwen3-30b-a3b": 0, "qwen3-235b-a22b": 0,
}


def _load(model: str, task: str):
    path = CIRCUITS / f"dag_{model}_{task}.pt"
    if not path.exists():
        return None
    dag = torch.load(path, weights_only=False, map_location="cpu")
    if "W_softmax" not in dag or "act" not in dag:
        return None
    W = dag["W_softmax"].to(torch.float64)
    L, N, _, _ = W.shape
    s = torch.arange(L).view(-1, 1, 1, 1)
    r = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s < r).to(W.dtype)
    W_fwd = W * fwd
    out = W_fwd.abs().sum(dim=(2, 3)).numpy().reshape(-1)
    in_ = W_fwd.abs().sum(dim=(0, 1)).numpy().reshape(-1)
    act_LN = dag["act"].to(torch.float64).numpy()
    act = act_LN.reshape(-1)
    depth = np.repeat(np.arange(L), N).astype(np.float64)
    return {"out": out, "in": in_, "act": act, "act_LN": act_LN,
            "depth": depth, "L": L, "N": N}


def _se_mask(model: str, act_LN: np.ndarray, frac: float = 0.75) -> np.ndarray:
    """Su-exact SE indicator (same logic as scatter_out_vs_act.py)."""
    L, N = act_LN.shape
    nd = NUM_DENSE.get(model, 0)
    upto = max(0, min(L, round((L + nd) * frac) - nd))
    subset = act_LN[:upto].reshape(-1)
    if subset.size == 0:
        return np.zeros(L * N, dtype=bool)
    thr = max(np.percentile(subset, 99.5), np.max(subset) // 10)
    mask = np.zeros((L, N), dtype=bool)
    mask[:upto] = act_LN[:upto] > thr
    return mask.reshape(-1)


def _pct(values: np.ndarray, v: float) -> float:
    """Percentage of entries strictly below v."""
    return 100.0 * float(np.mean(values < v))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="c4")
    p.add_argument("--top-k", type=int, default=5,
                   help="Set size for Influential (top-out) and Sensitive "
                        "(top-in) sets (default 5, matching the paper table).")
    p.add_argument("--include-layers", type=float, default=0.75,
                   help="Su's include_layers fraction for the SE set.")
    args = p.parse_args()

    cat_residuals: dict[str, list[float]] = {"I-only": [], "S-only": [],
                                             "I&S": [], "N-only": []}

    for m in MODELS:
        d = _load(m, args.task)
        if d is None:
            print(f"\n[{m}] MISSING")
            continue
        L, N = d["L"], d["N"]
        nd = NUM_DENSE.get(m, 0)
        out, in_, act, depth = d["out"], d["in"], d["act"], d["depth"]
        V = out.size

        # ---- Sets --------------------------------------------------------
        I_set = set(np.argsort(-out)[:args.top_k].tolist())
        N_set = set(np.argsort(-in_)[:args.top_k].tolist())
        S_set = set(np.flatnonzero(_se_mask(m, d["act_LN"], args.include_layers)).tolist())
        union = sorted(I_set | S_set | N_set)

        # ---- Step 2: per-model OLS  log10(out) ~ log10(act) + depth -------
        ok = (out > 0) & (act > 0)
        X = np.column_stack([np.log10(act[ok]),
                             depth[ok] / max(1, L - 1),
                             np.ones(ok.sum())])
        y = np.log10(out[ok])
        coef, *_ = np.linalg.lstsq(X, y, rcond=None)
        y_hat = X @ coef
        ss_res = float(np.sum((y - y_hat) ** 2))
        ss_tot = float(np.sum((y - np.mean(y)) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        # Residual lookup for every vertex (NaN where fit undefined).
        resid = np.full(V, np.nan)
        resid[ok] = y - y_hat

        print(f"\n{'=' * 100}")
        print(f"[{m}]  L={L} (+{nd} dense), N={N}, V={V}   "
              f"|I|={len(I_set)} |S|={len(S_set)} |N|={len(N_set)} "
              f"union={len(union)}")
        print(f"  OLS log10(out) = {coef[0]:+.3f}*log10(act) "
              f"{coef[1]:+.3f}*depth_frac {coef[2]:+.3f}   (R^2={r2:.3f}, "
              f"n={int(ok.sum())})")
        print(f"  {'label':>10s} {'dag_l':>5s} {'exp':>4s}  {'sets':<6s} "
              f"{'act':>9s} {'act%':>6s}  {'out':>9s} {'out%':>6s}  "
              f"{'in':>9s} {'in%':>6s}  {'resid':>7s}")

        for flat in union:
            l_dag, e = flat // N, flat % N
            label = f"L{l_dag + nd}E{e}"
            sets = []
            if flat in I_set:
                sets.append("I")
            if flat in S_set:
                sets.append("S")
            if flat in N_set:
                sets.append("N")
            sets_s = ",".join(sets)
            r = resid[flat]
            r_s = f"{r:+7.2f}" if np.isfinite(r) else "     --"
            print(f"  {label:>10s} {l_dag:>5d} {e:>4d}  {sets_s:<6s} "
                  f"{act[flat]:>9.4g} {_pct(act, act[flat]):>6.1f}  "
                  f"{out[flat]:>9.4g} {_pct(out, out[flat]):>6.1f}  "
                  f"{in_[flat]:>9.4g} {_pct(in_, in_[flat]):>6.1f}  {r_s}")

            # Category bookkeeping for the cross-model residual summary.
            if np.isfinite(r):
                in_I, in_S = flat in I_set, flat in S_set
                if in_I and in_S:
                    cat_residuals["I&S"].append(r)
                elif in_I:
                    cat_residuals["I-only"].append(r)
                elif in_S:
                    cat_residuals["S-only"].append(r)
                elif flat in N_set:
                    cat_residuals["N-only"].append(r)

    # ---- Cross-model residual summary (the hypothesis test) --------------
    print(f"\n{'=' * 100}")
    print("Residual summary by category (all models pooled)")
    print("Hypothesis: I-only >> 0,  S-only << 0,  I&S >= 0")
    print(f"  {'category':<8s} {'n':>4s} {'mean':>8s} {'median':>8s} "
          f"{'min':>8s} {'max':>8s}")
    for cat, vals in cat_residuals.items():
        if not vals:
            print(f"  {cat:<8s} {0:>4d}       --       --       --       --")
            continue
        a = np.array(vals)
        print(f"  {cat:<8s} {a.size:>4d} {a.mean():>+8.2f} "
              f"{np.median(a):>+8.2f} {a.min():>+8.2f} {a.max():>+8.2f}")


if __name__ == "__main__":
    main()
