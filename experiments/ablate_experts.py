"""Step B: ground-truth expert ablation, mirroring Su et al.'s procedure.

Ablation operator (theirs, reproduced exactly): the expert's down-projection
output is zeroed -- equivalent to their zeroing of the down_proj / w3 weight
tensors (Super-Experts-Profilling eval_utils.py, _load_layer_weight). The
router is untouched: the expert keeps its top-K slot and gating weight, but
contributes a zero vector. NO routing redistribution.

Evaluation protocol (theirs, reproduced exactly):
  - c4 validation (en/c4-validation.00000-of-00008.json.gz)
  - 256 random 2048-token windows: random doc with > seqlen tokens, random
    token offset (their get_c4 eval_mode, random.seed(0))
  - batch size 1; PPL = exp(mean over sequences of per-sequence mean NLL)

Reported per run: PPL (their metric). Additionally (our extension, clearly
separate): attention-sink share and position-0 massive-activation magnitude
on the first --n-sink-seqs windows.

Targets are single experts by default (expert sets via '+' are supported for
later, mirroring their --prune_experts semicolon lists).

Supported: olmoe, mixtral-*, phi-3.5-moe, qwen3-* (DeepSeek deferred).

Results merge into {result_path}/circuits/ablation_{model}_c4.json
(restart-safe; cached runs are skipped).

Usage (cluster, 1 GPU for olmoe / 4 for mixtral-8x7b):
    python experiments/ablate_experts.py --model olmoe
    python experiments/ablate_experts.py --model mixtral-8x7b
"""
from __future__ import annotations

import argparse
import json
import random
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
              "n_layers": 16, "experts_path": "mlp.experts",
              "down_attr": "down_proj", "num_dense": 0},
    "mixtral-8x7b": {"id": "mistralai/Mixtral-8x7B-v0.1", "n_experts": 8,
                     "n_layers": 32,
                     "experts_path": "block_sparse_moe.experts",
                     "down_attr": "w2", "num_dense": 0},
    "mixtral-8x22b": {"id": "mistralai/Mixtral-8x22B-v0.1", "n_experts": 8,
                      "n_layers": 56,
                      "experts_path": "block_sparse_moe.experts",
                      "down_attr": "w2", "num_dense": 0},
    "phi-3.5-moe": {"id": "microsoft/Phi-3.5-MoE-instruct", "n_experts": 16,
                    "n_layers": 32,
                    "experts_path": "block_sparse_moe.experts",
                    "down_attr": "w2", "num_dense": 0},
    "qwen3-30b-a3b": {"id": "Qwen/Qwen3-30B-A3B", "n_experts": 128,
                      "n_layers": 48, "experts_path": "mlp.experts",
                      "down_attr": "down_proj", "num_dense": 0},
    "qwen3-235b-a22b": {"id": "Qwen/Qwen3-235B-A22B", "n_experts": 128,
                        "n_layers": 94, "experts_path": "mlp.experts",
                        "down_attr": "down_proj", "num_dense": 0},
}

# Single-expert FP/FN candidates from the token/cross-rank/archetype
# analyses ('+' joint sets deliberately excluded for now).
DEFAULT_TARGETS = {
    "olmoe": "L1E9;L1E18;L2E30;L3E39;L4E14;L4E27;L9E8;L0E38;L6E4;L8E22",
    "mixtral-8x7b": "L1E3;L17E0;L18E5;L19E1;L19E6;L30E4",
    "phi-3.5-moe": "L0E6;L1E0;L3E3;L3E7;L5E10;L23E3;L27E9",
    "qwen3-30b-a3b": "L0E106;L1E68;L2E92;L3E82;L3E107;L21E69;L22E92;L33E69",
}

SEQLEN = 2048
N_SEQS = 256


def parse_targets(spec: str, num_dense: int) -> list[list[tuple[int, int]]]:
    """'L4E27;L1E18+L2E30' -> [[(4, 27)], [(1, 18), (2, 30)]]."""
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


class ZeroExpertOutput:
    """Context manager reproducing Su et al.'s pruning: forward hooks on the
    target experts' down-projection modules return zeros. Numerically
    identical to zeroing the down_proj/w3 weights (their implementation):
    the expert stays selected with its normal gating weight but contributes
    a zero vector."""

    def __init__(self, model, experts_path: str, down_attr: str,
                 group: list[tuple[int, int]]):
        self.model = model
        self.experts_path = experts_path
        self.down_attr = down_attr
        self.group = group
        self.handles = []

    def __enter__(self):
        layers = self.model.model.layers
        for l, e in self.group:
            experts = attrgetter(self.experts_path)(layers[l])
            down = getattr(experts[e], self.down_attr)

            def hook(_m, _inp, out):
                return torch.zeros_like(out)

            self.handles.append(down.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for h in self.handles:
            h.remove()
        return False


def su_c4_eval_windows(tok, n_seqs: int, seqlen: int) -> list[torch.Tensor]:
    """Su et al.'s get_c4(eval_mode=True): 256 random seqlen-token windows
    from c4 validation docs longer than seqlen, random.seed(0)."""
    import datasets
    valdata = datasets.load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation")
    random.seed(0)
    windows = []
    while len(windows) < n_seqs:
        while True:
            i = random.randint(0, len(valdata) - 1)
            enc = tok(valdata[i]["text"], return_tensors="pt")
            if enc.input_ids.shape[1] > seqlen:
                break
        i = random.randint(0, enc.input_ids.shape[1] - seqlen - 1)
        windows.append(enc.input_ids[0, i:i + seqlen])
    return windows


@torch.no_grad()
def eval_ppl(model, windows: list[torch.Tensor], batch_size: int = 1) -> float:
    """Su et al.'s PPL: exp(mean of per-sequence mean NLL). Batching is a
    pure speed optimisation -- the per-sequence NLL is computed explicitly,
    so any --batch-size gives numbers identical to their batch-1 loop."""
    from tqdm import tqdm
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    nlls: list[float] = []
    for i in tqdm(range(0, len(windows), batch_size), desc="ppl", leave=False):
        ids = torch.stack(windows[i:i + batch_size]).to(model.device)
        logits = model(ids).logits
        shift_logits = logits[:, :-1, :].permute(0, 2, 1)
        shift_labels = ids[:, 1:]
        loss = loss_fct(shift_logits, shift_labels)      # [B, T-1]
        nlls.extend(loss.float().mean(dim=1).tolist())   # per-sequence mean
    return float(np.exp(np.mean(nlls)))


@torch.no_grad()
def eval_sink(model, sink_ids: torch.Tensor) -> dict:
    out = model(sink_ids.to(model.device), output_attentions=True,
                output_hidden_states=True)
    att = torch.stack([a[:, :, 1:, 0].mean() for a in out.attentions])
    h0 = max(float(h[:, 0, :].abs().max()) for h in out.hidden_states)
    return {"att_sink": float(att.mean()), "max_h0": h0}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True, choices=list(MODELS))
    p.add_argument("--experts", default=None,
                   help="Ablation spec 'L4E27;L4E14;...' (default = curated "
                        "candidates, see DEFAULT_TARGETS).")
    p.add_argument("--random-controls", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seqs", type=int, default=N_SEQS)
    p.add_argument("--seq-len", type=int, default=SEQLEN)
    p.add_argument("--batch-size", type=int, default=8,
                   help="PPL eval batch size; numerically identical to Su's "
                        "batch-1 loop (per-sequence NLL computed explicitly).")
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
    controls: list[tuple[int, int]] = []
    while len(controls) < args.random_controls:
        l = int(rng.integers(0, cfg["n_layers"] - nd))
        e = int(rng.integers(0, cfg["n_experts"]))
        if (l, e) not in named and (l, e) not in controls:
            controls.append((l, e))
    runs += [[c] for c in controls]

    if not torch.cuda.is_available():
        sys.exit("No CUDA device visible -- device_map='auto' would fall "
                 "back to CPU and each PPL pass would take hours. Run on a "
                 "GPU node (or a torch build with CUDA).")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(cfg["id"])
    # sdpa for the (many) PPL forwards; the sink pass requests
    # output_attentions=True, for which transformers falls back to eager
    # attention on that call automatically.
    model = AutoModelForCausalLM.from_pretrained(
        cfg["id"], torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa").eval()
    print(f"device map: {getattr(model, 'hf_device_map', 'single device')}")

    windows = su_c4_eval_windows(tok, args.n_seqs, args.seq_len)
    sink_ids = torch.stack(windows[:args.n_sink_seqs])
    print(f"[{args.model}] {len(windows)} windows x {args.seq_len} tok "
          f"(Su et al. protocol); {len(runs)} ablation runs "
          f"({args.random_controls} random controls)")

    out_path = CIRCUITS / f"ablation_{args.model}_c4.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    if "baseline" not in results:
        ppl0 = eval_ppl(model, windows)
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
        with ZeroExpertOutput(model, cfg["experts_path"], cfg["down_attr"],
                              group):
            ppl = eval_ppl(model, windows)
            sink = eval_sink(model, sink_ids)
        results[lab] = {"ppl": ppl, **sink, "control": lab in is_control}
        out_path.write_text(json.dumps(results, indent=2))
        print(f"{lab}{' [ctrl]' if lab in is_control else ''}: "
              f"ppl={ppl:.3f} (x{ppl / base['ppl']:.3f})  "
              f"att_sink={sink['att_sink']:.4f} (base {base['att_sink']:.4f})  "
              f"max_h0={sink['max_h0']:.4g} (base {base['max_h0']:.4g})")

    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
