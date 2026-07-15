"""Router-alignment score per expert, computed from weights alone (step 3).

Hypothesis under test: an expert's routing influence out(v) decomposes as
  magnitude (act) x router-alignment x depth-headroom.
This script measures the alignment factor directly from checkpoints:

  For expert v at MoE layer s with down-projection W_v [d, m], model the
  expert's output energy distribution over residual-stream directions by
  its (uncentred) covariance W_v W_v^T. For each downstream MoE layer r,
  the router reads score directions g_i (rows of the gate matrix,
  optionally folded with the receiver's post-attention RMSNorm weight,
  then unit-normalised). Define

    align(v, r) = mean_i  g_i^T W_v W_v^T g_i / ||W_v||_F^2
                = fraction of v's output energy visible to r's router,
                  averaged over r's gate directions.

  align_raw(v) = mean over all downstream MoE layers r > s of align(v, r).
  For a random direction g the expectation is 1/d, so we report
  align_gain(v) = d * align_raw(v):   1.0 = random, >1 = router-aligned.

No forward passes: gate / down_proj / norm tensors are streamed directly
from the cached safetensors shards (local_files_only), so this runs on a
single GPU (recommended for qwen3-*/deepseek-v2) or CPU (small models).

Outputs {result_path}/circuits/router_alignment_{model}.pt with
  {"align_raw": [L, N], "align_gain": [L, N], "d_model": d,
   "moe_layers": [...], "fold_norm": bool}
(final DAG layer has no downstream MoE layer -> NaN).

The report step joins align_gain against the DAG features (out, act, in),
reprints the cross-rank union table with alignment columns, and computes
Spearman(gain, resid) where resid is the log10(out) ~ log10(act) + depth
residual from cross_rank_analysis. Pass/fail cases to eyeball:
  (a) OLMoE: gain(L4E14) >> gain(L4E27)      [DAG layers 4/4, experts 14/27]
  (b) Qwen3-30B mid-depth hubs (L21E69, L22E92, L33E69): high gain
  (c) DeepSeek-V2: SEs (L18E96, L21E94) low gain vs its top-out experts

Usage (cluster):
    python experiments/router_alignment.py --models olmoe
    python experiments/router_alignment.py                      # all models
    python experiments/router_alignment.py --report-only        # join only
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.cross_rank_analysis import (  # noqa: E402
    NUM_DENSE, _load as _load_dag_features, _se_mask,
)

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
CIRCUITS = Path(CFG["result_path"]) / "circuits"

# Tensor-name templates per model ({l} = model layer index, {e} = expert).
# gate rows are the router score directions; norm is the receiver-side
# post-attention RMSNorm weight folded into the gate read-out.
MODELS = {
    "mixtral-8x7b": {
        "id": "mistralai/Mixtral-8x7B-v0.1",
        "moe_layers": list(range(32)), "n_experts": 8,
        "gate": "model.layers.{l}.block_sparse_moe.gate.weight",
        "down": "model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight",
        "norm": "model.layers.{l}.post_attention_layernorm.weight",
    },
    "mixtral-8x22b": {
        "id": "mistralai/Mixtral-8x22B-v0.1",
        "moe_layers": list(range(56)), "n_experts": 8,
        "gate": "model.layers.{l}.block_sparse_moe.gate.weight",
        "down": "model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight",
        "norm": "model.layers.{l}.post_attention_layernorm.weight",
    },
    "phi-3.5-moe": {
        "id": "microsoft/Phi-3.5-MoE-instruct",
        "moe_layers": list(range(32)), "n_experts": 16,
        "gate": "model.layers.{l}.block_sparse_moe.gate.weight",
        "down": "model.layers.{l}.block_sparse_moe.experts.{e}.w2.weight",
        "norm": "model.layers.{l}.post_attention_layernorm.weight",
    },
    "deepseek-v2-lite": {
        "id": "deepseek-ai/DeepSeek-V2-Lite",
        "moe_layers": list(range(1, 27)), "n_experts": 64,
        "gate": "model.layers.{l}.mlp.gate.weight",
        "down": "model.layers.{l}.mlp.experts.{e}.down_proj.weight",
        "norm": "model.layers.{l}.post_attention_layernorm.weight",
    },
    "olmoe": {
        "id": "allenai/OLMoE-1B-7B-0924",
        "moe_layers": list(range(16)), "n_experts": 64,
        "gate": "model.layers.{l}.mlp.gate.weight",
        "down": "model.layers.{l}.mlp.experts.{e}.down_proj.weight",
        "norm": "model.layers.{l}.post_attention_layernorm.weight",
    },
    "qwen3-30b-a3b": {
        "id": "Qwen/Qwen3-30B-A3B",
        "moe_layers": list(range(48)), "n_experts": 128,
        "gate": "model.layers.{l}.mlp.gate.weight",
        "down": "model.layers.{l}.mlp.experts.{e}.down_proj.weight",
        "norm": "model.layers.{l}.post_attention_layernorm.weight",
    },
    "qwen3-235b-a22b": {
        "id": "Qwen/Qwen3-235B-A22B",
        "moe_layers": list(range(94)), "n_experts": 128,
        "gate": "model.layers.{l}.mlp.gate.weight",
        "down": "model.layers.{l}.mlp.experts.{e}.down_proj.weight",
        "norm": "model.layers.{l}.post_attention_layernorm.weight",
    },
    "deepseek-v2": {
        "id": "deepseek-ai/DeepSeek-V2",
        "moe_layers": list(range(1, 60)), "n_experts": 160,
        "gate": "model.layers.{l}.mlp.gate.weight",
        "down": "model.layers.{l}.mlp.experts.{e}.down_proj.weight",
        "norm": "model.layers.{l}.post_attention_layernorm.weight",
    },
}


# ---------------------------------------------------------------------------
# Weight streaming (no model instantiation).
# ---------------------------------------------------------------------------
class ShardReader:
    """Random access to tensors in a cached HF safetensors checkpoint."""

    def __init__(self, repo_id: str):
        from huggingface_hub import snapshot_download
        self.snap = Path(snapshot_download(repo_id, local_files_only=True))
        index_path = self.snap / "model.safetensors.index.json"
        if index_path.exists():
            with open(index_path) as f:
                self.weight_map = json.load(f)["weight_map"]
        else:
            # Single-file checkpoint: every tensor lives in model.safetensors.
            self.weight_map = None
        self._handles: dict[str, object] = {}

    def _open(self, shard: str):
        if shard not in self._handles:
            from safetensors import safe_open
            self._handles[shard] = safe_open(
                str(self.snap / shard), framework="pt", device="cpu")
        return self._handles[shard]

    def get(self, name: str) -> torch.Tensor:
        shard = (self.weight_map[name] if self.weight_map is not None
                 else "model.safetensors")
        return self._open(shard).get_tensor(name)


# ---------------------------------------------------------------------------
# Alignment computation.
# ---------------------------------------------------------------------------
def compute_alignment(model: str, device: torch.device, fold_norm: bool,
                      chunk: int = 8) -> dict:
    cfg = MODELS[model]
    reader = ShardReader(cfg["id"])
    moe_layers = cfg["moe_layers"]
    L, N = len(moe_layers), cfg["n_experts"]

    # Unit-normalised (optionally norm-folded) gate rows per DAG layer.
    def gate_rows(dag_l: int) -> torch.Tensor:
        ml = moe_layers[dag_l]
        G = reader.get(cfg["gate"].format(l=ml)).to(device, torch.float32)
        if fold_norm:
            gamma = reader.get(cfg["norm"].format(l=ml)).to(device, torch.float32)
            G = G * gamma.unsqueeze(0)
        return G / G.norm(dim=1, keepdim=True).clamp(min=1e-12)

    d = gate_rows(0).shape[1]
    align_raw = torch.full((L, N), float("nan"))

    # Iterate sender layers back-to-front, maintaining the suffix sum of the
    # per-layer gate Grams S_r = mean_i g_i g_i^T, so at sender s the
    # accumulator equals sum_{r > s} S_r without storing all L matrices.
    suffix = torch.zeros((d, d), dtype=torch.float32, device=device)
    n_down = 0
    for s in range(L - 1, -1, -1):
        if n_down > 0:
            S_bar = suffix / n_down
            for e0 in range(0, N, chunk):
                e1 = min(e0 + chunk, N)
                Ws = [reader.get(cfg["down"].format(l=moe_layers[s], e=e))
                      .to(device, torch.float32) for e in range(e0, e1)]
                W = torch.stack(Ws)                     # [k, d, m]
                SW = torch.matmul(S_bar, W)             # [k, d, m]
                num = (W * SW).sum(dim=(1, 2))          # tr(W^T S_bar W)
                den = (W * W).sum(dim=(1, 2)).clamp(min=1e-12)
                align_raw[s, e0:e1] = (num / den).cpu()
        G = gate_rows(s)
        suffix += G.T @ G / G.shape[0]
        n_down += 1
        print(f"  [{model}] layer {s}/{L - 1} done", flush=True)

    return {"align_raw": align_raw, "align_gain": align_raw * d,
            "d_model": d, "moe_layers": moe_layers, "fold_norm": fold_norm}


# ---------------------------------------------------------------------------
# Report: join alignment against DAG features.
# ---------------------------------------------------------------------------
def report(model: str, task: str, top_k: int, include_layers: float) -> dict | None:
    apath = CIRCUITS / f"router_alignment_{model}.pt"
    if not apath.exists():
        print(f"\n[{model}] no alignment file ({apath})")
        return None
    A = torch.load(apath, weights_only=False, map_location="cpu")
    gain = A["align_gain"].numpy().reshape(-1)

    d = _load_dag_features(model, task)
    if d is None:
        print(f"\n[{model}] alignment computed but DAG missing; skipping join")
        return None
    L, N = d["L"], d["N"]
    nd = NUM_DENSE.get(model, 0)
    out, in_, act, depth = d["out"], d["in"], d["act"], d["depth"]
    V = out.size

    I_set = set(np.argsort(-out)[:top_k].tolist())
    N_set = set(np.argsort(-in_)[:top_k].tolist())
    S_set = set(np.flatnonzero(_se_mask(model, d["act_LN"], include_layers)).tolist())
    union = sorted(I_set | S_set | N_set)

    # Same OLS as cross_rank_analysis: resid = alignment-shaped unknown.
    ok = (out > 0) & (act > 0)
    X = np.column_stack([np.log10(act[ok]), depth[ok] / max(1, L - 1),
                         np.ones(ok.sum())])
    y = np.log10(out[ok])
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = np.full(V, np.nan)
    resid[ok] = y - X @ coef

    both = ok & np.isfinite(gain)
    from scipy.stats import spearmanr
    rho_gain_out, _ = spearmanr(gain[both], out[both])
    rho_gain_resid, _ = spearmanr(gain[both], resid[both])

    def pct(vals, v):
        return 100.0 * float(np.mean(vals < v))

    finite_gain = gain[np.isfinite(gain)]
    print(f"\n{'=' * 96}")
    print(f"[{model}]  d={A['d_model']}, fold_norm={A['fold_norm']}   "
          f"Spearman(gain, out)={rho_gain_out:+.3f}   "
          f"Spearman(gain, resid)={rho_gain_resid:+.3f}")
    print(f"  {'label':>10s}  {'sets':<6s} {'act%':>6s} {'out%':>6s} "
          f"{'in%':>6s}  {'gain':>7s} {'gain%':>6s}  {'resid':>7s}")
    rows = []
    for flat in union:
        l_dag, e = flat // N, flat % N
        sets = ",".join(c for c, in_set in
                        [("I", flat in I_set), ("S", flat in S_set),
                         ("N", flat in N_set)] if in_set)
        g = gain[flat]
        g_s = f"{g:7.2f}" if np.isfinite(g) else "     --"
        gp_s = f"{pct(finite_gain, g):6.1f}" if np.isfinite(g) else "    --"
        r = resid[flat]
        r_s = f"{r:+7.2f}" if np.isfinite(r) else "     --"
        print(f"  {'L' + str(l_dag + nd) + 'E' + str(e):>10s}  {sets:<6s} "
              f"{pct(act, act[flat]):>6.1f} {pct(out, out[flat]):>6.1f} "
              f"{pct(in_, in_[flat]):>6.1f}  {g_s} {gp_s}  {r_s}")
        rows.append({"flat": flat, "sets": sets, "gain": g, "resid": r})
    return {"rho_gain_out": rho_gain_out, "rho_gain_resid": rho_gain_resid,
            "rows": rows}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=list(MODELS),
                   choices=list(MODELS))
    p.add_argument("--task", default="c4")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--include-layers", type=float, default=0.75)
    p.add_argument("--no-fold-norm", action="store_true",
                   help="Skip folding the receiver post-attention norm weight "
                        "into the gate directions.")
    p.add_argument("--chunk", type=int, default=8,
                   help="Experts per GEMM batch (reduce if OOM).")
    p.add_argument("--report-only", action="store_true",
                   help="Skip computation; join existing alignment files "
                        "against DAG features.")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not args.report_only:
        print(f"Computing alignment on {device}")
        for m in args.models:
            out_path = CIRCUITS / f"router_alignment_{m}.pt"
            if out_path.exists():
                print(f"[{m}] exists, skipping ({out_path})")
                continue
            res = compute_alignment(m, device, fold_norm=not args.no_fold_norm,
                                    chunk=args.chunk)
            torch.save(res, out_path)
            print(f"[{m}] saved {out_path}")

    for m in args.models:
        report(m, args.task, args.top_k, args.include_layers)


if __name__ == "__main__":
    main()
