"""Step B: ground-truth expert ablation — PPL + attention-sink damage.

For each ablation target (single expert or joint set), mask the expert(s)
out of routing — the router logit is set to -inf BEFORE top-K selection, so
the renormalised top-K redistributes to other experts (deployment-style
expert removal, comparable to Su et al.'s pruning) — then measure on c4:

  - PPL over --n-seqs held-out documents (vs. un-ablated baseline)
  - sink metrics on --n-sink-seqs documents:
      att_sink : mean attention mass from queries (pos >= 1) to position 0,
                 averaged over layers x heads (Su's mechanism variable)
      max_h0   : max |hidden state| at position 0 across layers
                 (massive-activation indicator)

Causal labels this produces:
  FP of Su's criterion  = SE-flagged expert, small dPPL, no sink collapse
  FN of Su's criterion  = unflagged expert, large dPPL

Target syntax (model-absolute LxEy labels, as in the paper tables):
  --experts "L4E27;L4E14;L1E18+L2E30+L3E39+L9E8"
  ';' separates independent ablation runs, '+' ablates a set jointly
  (the joint run tests redundancy of the sink infrastructure).

Supported models: linear-gate architectures (olmoe, mixtral-*, phi-3.5-moe,
qwen3-*). DeepSeek's custom MoEGate returns indices rather than logits and
needs a different hook — deferred.

Results are merged into {result_path}/circuits/ablation_{model}_{task}.json
(restart-safe: existing entries are kept, so runs can be split).

Usage (cluster, 1 GPU):
    python experiments/ablate_experts.py --model olmoe            # defaults
    python experiments/ablate_experts.py --model olmoe \
        --experts "L4E27;L4E14" --random-controls 5
"""
from __future__ import annotations

import argparse
import json
import sys
from operator import attrgetter
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)
CIRCUITS = Path(CFG["result_path"]) / "circuits"

MODELS = {
    "olmoe": {"id": "allenai/OLMoE-1B-7B-0924", "n_experts": 64,
              "n_layers": 16, "gate_path": "mlp.gate", "num_dense": 0},
    "mixtral-8x7b": {"id": "mistralai/Mixtral-8x7B-v0.1", "n_experts": 8,
                     "n_layers": 32, "gate_path": "block_sparse_moe.gate",
                     "num_dense": 0},
    "mixtral-8x22b": {"id": "mistralai/Mixtral-8x22B-v0.1", "n_experts": 8,
                      "n_layers": 56, "gate_path": "block_sparse_moe.gate",
                      "num_dense": 0},
    "phi-3.5-moe": {"id": "microsoft/Phi-3.5-MoE-instruct", "n_experts": 16,
                    "n_layers": 32, "gate_path": "block_sparse_moe.gate",
                    "num_dense": 0},
    "qwen3-30b-a3b": {"id": "Qwen/Qwen3-30B-A3B", "n_experts": 128,
                      "n_layers": 48, "gate_path": "mlp.gate", "num_dense": 0},
    "qwen3-235b-a22b": {"id": "Qwen/Qwen3-235B-A22B", "n_experts": 128,
                        "n_layers": 94, "gate_path": "mlp.gate",
                        "num_dense": 0},
}

# Default targets: our FP/FN candidates from the token/cross-rank analyses,
# plus the joint redundant-sink set for OLMoE.
DEFAULT_TARGETS = {
    "olmoe": ("L1E9;L1E18;L2E30;L3E39;L4E14;L4E27;L9E8;"
              "L1E18+L2E30+L3E39+L9E8"),
    "qwen3-30b-a3b": "L1E68;L2E92;L3E82;L3E107;L21E69;L22E92;L33E69",
    "mixtral-8x7b": "L1E3;L17E0;L18E5;L19E1;L19E6;L30E4",
}


def parse_targets(spec: str, num_dense: int) -> list[list[tuple[int, int]]]:
    """'L4E27;L1E18+L2E30' -> [[(4, 27)], [(1, 18), (2, 30)]] with
    model-absolute layer converted to module-list index (minus num_dense —
    identical for all currently supported models)."""
    runs = []
    for run in spec.split(";"):
        group = []
        for lab in run.strip().split("+"):
            lab = lab.strip().upper()
            l_str, e_str = lab.lstrip("L").split("E")
            group.append((int(l_str) - num_dense, int(e_str)))
        runs.append(group)
    return runs


def label_of(group: list[tuple[int, int]], num_dense: int) -> str:
    return "+".join(f"L{l + num_dense}E{e}" for l, e in group)


class RoutingMask:
    """Context manager: forward hooks on the gate Linear of the target
    layers that set the target experts' logits to dtype-min before top-K."""

    def __init__(self, model, gate_path: str, group: list[tuple[int, int]]):
        self.handles = []
        by_layer: dict[int, list[int]] = {}
        for l, e in group:
            by_layer.setdefault(l, []).append(e)
        self.by_layer = by_layer
        self.model = model
        self.gate_path = gate_path

    def __enter__(self):
        layers = self.model.model.layers
        for l, experts in self.by_layer.items():
            gate = attrgetter(self.gate_path)(layers[l])
            idx = torch.tensor(experts)

            def hook(_m, _inp, out, _idx=idx):
                out[..., _idx] = torch.finfo(out.dtype).min
                return out

            self.handles.append(gate.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False


@torch.no_grad()
def eval_ppl(model, batches: list[torch.Tensor]) -> float:
    losses = []
    for ids in batches:
        out = model(ids.to(model.device), labels=ids.to(model.device))
        losses.append(float(out.loss))
    return float(np.exp(np.mean(losses)))


@torch.no_grad()
def eval_sink(model, sink_ids: torch.Tensor) -> dict:
    out = model(sink_ids.to(model.device), output_attentions=True,
                output_hidden_states=True)
    # Mean attention mass to position 0 from all later queries, over
    # layers x heads x batch.
    att = torch.stack([a[:, :, 1:, 0].mean() for a in out.attentions])
    # Max |hidden| at position 0 across layers (massive-activation site).
    h0 = max(float(h[:, 0, :].abs().max()) for h in out.hidden_states)
    return {"att_sink": float(att.mean()), "max_h0": h0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--task", default="c4")
    p.add_argument("--experts", default=None,
                   help="Ablation spec; default = curated candidate list "
                        "for the model (see DEFAULT_TARGETS).")
    p.add_argument("--random-controls", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seqs", type=int, default=200)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--n-sink-seqs", type=int, default=8)
    args = p.parse_args()

    cfg = MODELS[args.model]
    nd = cfg["num_dense"]
    spec = args.experts or DEFAULT_TARGETS.get(args.model)
    if spec is None:
        p.error(f"no default targets for {args.model}; pass --experts")
    runs = parse_targets(spec, nd)

    rng = np.random.default_rng(args.seed)
    named = {(l, e) for grp in runs for l, e in grp}
    controls = []
    while len(controls) < args.random_controls:
        l = int(rng.integers(0, cfg["n_layers"] - nd))
        e = int(rng.integers(0, cfg["n_experts"]))
        if (l, e) not in named and (l, e) not in controls:
            controls.append((l, e))
    runs += [[c] for c in controls]

    if args.task != "c4":
        raise NotImplementedError("only c4 wired up for now")
    from dataset.c4_dataset import c4_dataset_helper
    texts = c4_dataset_helper(dataset_len=args.n_seqs,
                              min_words=args.seq_len)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["id"])
    model = AutoModelForCausalLM.from_pretrained(
        cfg["id"], torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="eager").eval()

    enc = [tok(t, return_tensors="pt", truncation=True,
               max_length=args.seq_len).input_ids[0] for t in texts]
    enc = [e for e in enc if e.numel() == args.seq_len]
    batches = [torch.stack(enc[i:i + args.batch_size])
               for i in range(0, len(enc) - args.batch_size + 1,
                              args.batch_size)]
    sink_ids = torch.stack(enc[:args.n_sink_seqs])
    print(f"[{args.model}] {len(enc)} seqs x {args.seq_len} tok, "
          f"{len(batches)} batches; {len(runs)} ablation runs "
          f"({args.random_controls} random controls)")

    out_path = CIRCUITS / f"ablation_{args.model}_{args.task}.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    if "baseline" not in results:
        ppl0 = eval_ppl(model, batches)
        sink0 = eval_sink(model, sink_ids)
        results["baseline"] = {"ppl": ppl0, **sink0}
        out_path.write_text(json.dumps(results, indent=2))
    base = results["baseline"]
    print(f"baseline: ppl={base['ppl']:.3f} att_sink={base['att_sink']:.4f} "
          f"max_h0={base['max_h0']:.4g}")

    is_control = {label_of([c], nd) for c in controls}
    for group in runs:
        lab = label_of(group, nd)
        if lab in results:
            print(f"{lab}: cached, skipping")
            continue
        with RoutingMask(model, cfg["gate_path"], group):
            ppl = eval_ppl(model, batches)
            sink = eval_sink(model, sink_ids)
        results[lab] = {"ppl": ppl, **sink,
                        "control": lab in is_control}
        out_path.write_text(json.dumps(results, indent=2))
        print(f"{lab}{' [ctrl]' if lab in is_control else ''}: "
              f"ppl={ppl:.3f} (x{ppl / base['ppl']:.3f})  "
              f"att_sink={sink['att_sink']:.4f} "
              f"(base {base['att_sink']:.4f})  "
              f"max_h0={sink['max_h0']:.4g} (base {base['max_h0']:.4g})")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
