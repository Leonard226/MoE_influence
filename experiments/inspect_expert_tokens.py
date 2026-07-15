"""Step A: token-level characterization of Influential / Super / Sensitive experts.

For every expert in the union of
    I = top-K by out(v),  S = Su-exact Super Experts,  N = top-K by in(v)
inspect the top-100 highest-routing-weight token events stored in the DAG
(top_weight / top_prompt / top_pos / top_token buffers) and summarise:

  - pos0% / pos<=3%: fraction of events at sink positions (start of prompt)
  - median position
  - token-class mix: special / whitespace / punct / numeric / content
  - the most frequent decoded tokens

Su et al.'s attention-sink mechanism predicts Super-Expert events concentrate
on early positions and special/punctuation tokens; "routing hub" experts
(high out, moderate act) should instead fire on content tokens spread across
positions. This gives the first ground-truth-free evidence for the FP/FN
hypothesis using data we already have -- no model weights needed (only the
tokenizer, a tiny download).

Usage:
    python experiments/inspect_expert_tokens.py
    python experiments/inspect_expert_tokens.py --models olmoe --top-show 15
"""
from __future__ import annotations

import argparse
import string
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from experiments.cross_rank_analysis import NUM_DENSE, _se_mask  # noqa: E402

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
CIRCUITS = Path(CFG["result_path"]) / "circuits"

MODELS = [
    "mixtral-8x7b", "mixtral-8x22b", "phi-3.5-moe",
    "deepseek-v2-lite", "olmoe",
    "qwen3-30b-a3b", "qwen3-235b-a22b", "deepseek-v2",
]

_PUNCT = set(string.punctuation) | set("··–—‘’“”…«»、。，！？；：")


def _classify(tid: int, decoded: str, special_ids: set[int]) -> str:
    if tid in special_ids:
        return "special"
    s = decoded.strip()
    if s == "":
        return "whitespace"
    if all(c in _PUNCT for c in s):
        return "punct"
    if s.isdigit():
        return "numeric"
    return "content"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    p.add_argument("--task", default="c4")
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--include-layers", type=float, default=0.75)
    p.add_argument("--top-show", type=int, default=10,
                   help="How many most-frequent tokens to print per expert.")
    args = p.parse_args()

    from transformers import AutoTokenizer

    # Pooled per-category aggregates across all models.
    agg: dict[str, list[tuple[float, float, float]]] = {}

    for m in args.models:
        path = CIRCUITS / f"dag_{m}_{args.task}.pt"
        if not path.exists():
            print(f"\n[{m}] MISSING ({path})")
            continue
        dag = torch.load(path, weights_only=False, map_location="cpu")
        W = dag["W_softmax"].to(torch.float64)
        L, N, _, _ = W.shape
        nd = NUM_DENSE.get(m, 0)
        s_idx = torch.arange(L).view(-1, 1, 1, 1)
        r_idx = torch.arange(L).view(1, 1, -1, 1)
        fwd = (s_idx < r_idx).to(W.dtype)
        W_fwd = W * fwd
        out = W_fwd.abs().sum(dim=(2, 3)).numpy().reshape(-1)
        in_ = W_fwd.abs().sum(dim=(0, 1)).numpy().reshape(-1)
        act_LN = dag["act"].to(torch.float64).numpy()
        act = act_LN.reshape(-1)

        I_set = set(np.argsort(-out)[:args.top_k].tolist())
        N_set = set(np.argsort(-in_)[:args.top_k].tolist())
        S_set = set(np.flatnonzero(
            _se_mask(m, act_LN, args.include_layers)).tolist())
        union = sorted(I_set | S_set | N_set)

        tok = AutoTokenizer.from_pretrained(dag["model"], trust_remote_code=True)
        special_ids = set(tok.all_special_ids)

        tw = dag["top_weight"]   # [L, N, K] float, empty slot = -1
        tp = dag["top_pos"]      # [L, N, K] int16
        tt = dag["top_token"]    # [L, N, K] int32

        print(f"\n{'=' * 100}")
        print(f"[{m}]  L={L} (+{nd} dense), N={N}, "
              f"max_tokens={dag.get('max_tokens', '?')}, "
              f"buffer=top-{dag.get('k_top_tokens', tw.shape[-1])} "
              f"routing-weight events per expert")
        print(f"  {'label':>10s}  {'sets':<6s} {'act%':>5s} {'out%':>5s}  "
              f"{'pos0%':>5s} {'pos<=3%':>7s} {'medpos':>6s}  "
              f"{'spec%':>5s} {'ws%':>4s} {'punct%':>6s} {'num%':>4s} "
              f"{'cont%':>5s}  top tokens")

        for flat in union:
            l_dag, e = flat // N, flat % N
            sets = ",".join(c for c, in_s in
                            [("I", flat in I_set), ("S", flat in S_set),
                             ("N", flat in N_set)] if in_s)
            w = tw[l_dag, e].numpy()
            valid = w >= 0
            n_ev = int(valid.sum())
            if n_ev == 0:
                print(f"  {'L' + str(l_dag + nd) + 'E' + str(e):>10s}  "
                      f"{sets:<6s}  (no recorded events)")
                continue
            pos = tp[l_dag, e].numpy()[valid].astype(int)
            tid = tt[l_dag, e].numpy()[valid].astype(int)

            decoded = [tok.decode([t]) for t in tid]
            classes = [_classify(t, s, special_ids)
                       for t, s in zip(tid, decoded)]
            cls_frac = {c: 100.0 * classes.count(c) / n_ev
                        for c in ["special", "whitespace", "punct",
                                  "numeric", "content"]}
            pos0 = 100.0 * float(np.mean(pos == 0))
            pos3 = 100.0 * float(np.mean(pos <= 3))
            medp = float(np.median(pos))

            top_toks = Counter(decoded).most_common(args.top_show)
            tok_str = " ".join(f"{s!r}x{c}" for s, c in top_toks)

            act_pct = 100.0 * float(np.mean(act < act[flat]))
            out_pct = 100.0 * float(np.mean(out < out[flat]))
            print(f"  {'L' + str(l_dag + nd) + 'E' + str(e):>10s}  {sets:<6s} "
                  f"{act_pct:>5.1f} {out_pct:>5.1f}  "
                  f"{pos0:>5.1f} {pos3:>7.1f} {medp:>6.0f}  "
                  f"{cls_frac['special']:>5.1f} {cls_frac['whitespace']:>4.1f} "
                  f"{cls_frac['punct']:>6.1f} {cls_frac['numeric']:>4.1f} "
                  f"{cls_frac['content']:>5.1f}  {tok_str}")

            cat = ("I&S" if flat in I_set and flat in S_set else
                   "I-only" if flat in I_set else
                   "S-only" if flat in S_set else "N-only")
            agg.setdefault(cat, []).append(
                (pos3, cls_frac["special"] + cls_frac["punct"]
                 + cls_frac["whitespace"], cls_frac["content"]))

    print(f"\n{'=' * 100}")
    print("Category aggregates (all models pooled; per-expert means)")
    print("Sink-mechanism prediction: S-only high pos<=3% and non-content%; "
          "I-only low pos<=3%, high content%.")
    print(f"  {'category':<8s} {'n':>3s} {'pos<=3%':>8s} "
          f"{'spec+punct+ws%':>15s} {'content%':>9s}")
    for cat in ["I&S", "I-only", "S-only", "N-only"]:
        if cat not in agg:
            continue
        a = np.array(agg[cat])
        print(f"  {cat:<8s} {len(a):>3d} {a[:, 0].mean():>8.1f} "
              f"{a[:, 1].mean():>15.1f} {a[:, 2].mean():>9.1f}")


if __name__ == "__main__":
    main()
