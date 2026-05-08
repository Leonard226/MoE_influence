"""Joint ablation of candidate circuits with selectivity test.

For each candidate circuit V_C, jointly zero the score contributions of all members
(via the score-decomposition trick — no actual model surgery) and measure the
downstream AARV. Compare to ablating size-matched random expert sets to control
for general expert importance.

Two regimes:
  (a) Trigger-defined circuits (hand-specified token predicate).
      We split tokens into trigger vs control and report selectivity and specificity.
      Currently:
          C1 = {M1E9, M4E14}, hypothesized determiner-detection chain.
          C2 = {M2E30},       hypothesized capitalization-inhibition source.
      The trigger predicate is on the current token (the score-subtraction
      ablation is per-position; the relevant trigger is where the circuit
      fires, which is the token itself).

  (b) Trigger-discovery circuits (no hand-specified predicate).
      Graph-discovered chains / fan-outs / communities. Trigger is unknown a
      priori, so we only report aggregate AARV vs random. We additionally
      accumulate per-(token_id) AARV across the corpus and print the top
      tokens by induced rank shift — this reveals the trigger class
      empirically.

Quantities reported per target:
    AARV(trigger), AARV(control)         — when trigger_fn is set.
    Selectivity = AARV(T) / AARV(C)      — when trigger_fn is set.
    AARV(all)                            — always.
    Specificity = AARV(real) / AARV(random) — always.
    Top-K tokens by induced AARV         — when trigger_fn is None.

Usage:
    python experiments/circuits/ablate_circuit.py --dataset c4

Output:
    {result_path}/circuits/ablation_{dataset}.json
    stdout summary.
"""
import argparse
import importlib
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
output_dir = os.path.join(config["result_path"], "circuits")
os.makedirs(output_dir, exist_ok=True)

DATASETS = {
    "c4":   ("dataset.c4_dataset",   "c4_dataset_helper"),
    "math": ("dataset.math_dataset", "open_r1_math_dataset_helper"),
    "code": ("dataset.code_dataset", "code_dataset_helper"),
}

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--dataset", choices=list(DATASETS), default="c4")
parser.add_argument("--n-prompts", type=int, default=5000)
parser.add_argument("--n-random-trials", type=int, default=5,
                    help="Number of random matched-layer ablations per circuit (default: 5).")
parser.add_argument("--seed", type=int, default=0,
                    help="Seed for both dataset shuffle (if any) and random trial generation.")
parser.add_argument("--top-tokens", type=int, default=30,
                    help="Top-K tokens by induced AARV to print for trigger-discovery circuits (default: 30).")
parser.add_argument("--min-token-count", type=int, default=20,
                    help="Min occurrences for a token id to be eligible for the top-K diagnostic (default: 20).")
args = parser.parse_args()

device = "cuda:0"
torch.set_default_device(device)
torch.set_grad_enabled(False)

from customized_models.modeling_olmoe_customized import OlmoeForCausalLM
from transformers import AutoTokenizer

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_LAYERS = 16
N_EXPERTS = 64
D_E = 2048
TOP_K = 8
EPS = 1e-5
N_PROMPTS = args.n_prompts
BSZ = 50
MAX_TOKENS = 32

# ---- Circuit definitions ----
# Each entry: members (list of (layer, expert_idx)) and trigger_fn (str or None).
# trigger_fn=None ⇒ trigger-discovery mode: print top tokens by induced AARV.
CIRCUITS = {
    "C1_determiner": {
        "members": [(1, 9), (4, 14)],
        "trigger_fn": "determiner",
        "description": "M1E9 → M4E14 chain hypothesized to detect/relay determiners",
    },
    "C2_capitalization": {
        "members": [(2, 30)],
        "trigger_fn": "capitalization",
        "description": "M2E30 hypothesized to inhibit late-layer experts on capitalized openers",
    },
    "C3_chain_M1E18_M2E30_M14E60": {
        "members": [(1, 18), (2, 30), (14, 60)],
        "trigger_fn": None,
        "description": "Top APS chain through M2E30 from discover_circuits.py",
    },
    "C4_chain_M0E6_M2E30": {
        "members": [(0, 6), (2, 30)],
        "trigger_fn": None,
        "description": "Top APS 2-step chain into M2E30",
    },
    "C5_chain_M0E6_M4E14": {
        "members": [(0, 6), (4, 14)],
        "trigger_fn": None,
        "description": "Top APS 2-step chain into M4E14 (alternative determiner relay)",
    },
    "C6_fan_out_M14E60": {
        "members": [(14, 60)],
        "trigger_fn": None,
        "description": "Top fan-out source from discover_circuits.py",
    },
}


def determiner_trigger_mask(input_ids, attention_mask, tokenizer):
    """Mask True for tokens whose CURRENT token decodes to 'a' or 'the' (case-insensitive).

    These are the positions where M1E9 z=915 fires (per Leonard's `MoEs/main.tex`),
    and where the analytical ablation should produce its largest rank shift.
    """
    bsz, n_tok = input_ids.shape
    mask = torch.zeros((bsz, n_tok), dtype=torch.bool, device=input_ids.device)
    for b in range(bsz):
        for t in range(n_tok):
            if not bool(attention_mask[b, t]):
                continue
            cur_id = int(input_ids[b, t].item())
            cur_text = tokenizer.decode([cur_id]).strip().lower()
            if cur_text in {"a", "the"}:
                mask[b, t] = True
    return mask


def capitalization_trigger_mask(input_ids, attention_mask, tokenizer):
    """Mask True for tokens whose CURRENT token's stripped text starts with an uppercase letter.

    These are the positions where M2E30 z=742 fires (per Leonard's `MoEs/main.tex`).
    """
    bsz, n_tok = input_ids.shape
    mask = torch.zeros((bsz, n_tok), dtype=torch.bool, device=input_ids.device)
    for b in range(bsz):
        for t in range(n_tok):
            if not bool(attention_mask[b, t]):
                continue
            cur_id = int(input_ids[b, t].item())
            cur_text = tokenizer.decode([cur_id]).strip()
            if cur_text and cur_text[0].isupper():
                mask[b, t] = True
    return mask


TRIGGER_FNS = {
    "determiner": determiner_trigger_mask,
    "capitalization": capitalization_trigger_mask,
}

# ---- Load model + tokenizer + dataset ----
print(f"Loading {MODEL_ID} ...", flush=True)
t0 = time.time()
model = OlmoeForCausalLM.from_pretrained(
    MODEL_ID, attn_implementation="eager", torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
VOCAB = len(tokenizer)
print(f"  loaded in {time.time() - t0:.1f}s; vocab={VOCAB}", flush=True)

G = torch.stack([
    model.model.layers[l].mlp.gate.weight.detach().to(device, dtype=torch.float32)
    for l in range(N_LAYERS)
])  # [L, n_experts, d_e]
gamma = torch.stack([
    model.model.layers[l].post_attention_layernorm.weight.detach().to(device, dtype=torch.float32)
    for l in range(N_LAYERS)
])  # [L, d_e]

mod_name, fn_name = DATASETS[args.dataset]
loader = getattr(importlib.import_module(mod_name), fn_name)
print(f"Loading dataset={args.dataset!r} ({N_PROMPTS} prompts) ...", flush=True)
t0 = time.time()
prompts = loader(dataset_len=N_PROMPTS, seed=None, min_words=MAX_TOKENS)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)


# ---- Random matched-layer baselines ----
rng = np.random.default_rng(args.seed)


def sample_random_circuit(members):
    """Same layers, different (random) experts not in V_C."""
    out = []
    forbidden_at = {c: set() for c, _ in members}
    for c, j in members:
        forbidden_at[c].add(j)
    for c, _ in members:
        candidates = [k for k in range(N_EXPERTS) if k not in forbidden_at[c]]
        chosen = int(rng.choice(candidates))
        forbidden_at[c].add(chosen)
        out.append((c, chosen))
    return out


random_circuits = {
    cname: [sample_random_circuit(cdef["members"]) for _ in range(args.n_random_trials)]
    for cname, cdef in CIRCUITS.items()
}

# All ablation targets: (label, circuit_name, members)
all_targets = []
for cname, cdef in CIRCUITS.items():
    all_targets.append((f"real_{cname}", cname, cdef["members"]))
    for trial_idx, rc in enumerate(random_circuits[cname]):
        all_targets.append((f"random_{cname}_t{trial_idx}", cname, rc))

print(f"\nAblation targets ({len(all_targets)}):")
for label, cname, members in all_targets:
    members_str = ", ".join(f"M{c}E{j}" for c, j in members)
    print(f"  {label:<40}  {{{members_str}}}")


# ---- Accumulators per (target, layer) ----
# Always: aggregate AARV per layer across all attended tokens.
# If has_trigger: also trigger/control split per layer.
# If per_token: also per-token-id sum/count of mean-downstream AARV (only for real_*).
def make_accum(has_trigger, per_token):
    a = {
        "all_sum":   torch.zeros(N_LAYERS, dtype=torch.float64, device=device),
        "all_count": torch.zeros(N_LAYERS, dtype=torch.long, device=device),
    }
    if has_trigger:
        a["trigger_sum"]   = torch.zeros(N_LAYERS, dtype=torch.float64, device=device)
        a["trigger_count"] = torch.zeros(N_LAYERS, dtype=torch.long, device=device)
        a["control_sum"]   = torch.zeros(N_LAYERS, dtype=torch.float64, device=device)
        a["control_count"] = torch.zeros(N_LAYERS, dtype=torch.long, device=device)
    if per_token:
        a["per_token_sum"]   = torch.zeros(VOCAB, dtype=torch.float64, device=device)
        a["per_token_count"] = torch.zeros(VOCAB, dtype=torch.long, device=device)
    return a


acc = {
    label: make_accum(
        has_trigger=CIRCUITS[cname]["trigger_fn"] is not None,
        per_token=label.startswith("real_"),
    )
    for label, cname, _ in all_targets
}

n_batches = (N_PROMPTS + BSZ - 1) // BSZ
print(f"\nRunning {n_batches} batches (bsz={BSZ}, max_tokens={MAX_TOKENS}) ...", flush=True)
t_start = time.time()

for B in range(0, N_PROMPTS, BSZ):
    batch = prompts[B:B + BSZ]
    inputs = tokenizer(batch, return_tensors="pt", padding=False, truncation=True, max_length=MAX_TOKENS)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    # ---- Trigger masks (per-token-class) ----
    tmasks = {}
    for tname in {cdef["trigger_fn"] for cdef in CIRCUITS.values()
                  if cdef["trigger_fn"] is not None}:
        tmasks[tname] = TRIGGER_FNS[tname](input_ids, attention_mask, tokenizer)

    # ---- Forward pass ----
    _, hook_dict = model(input_ids=input_ids, attention_mask=attention_mask)
    after_res1   = hook_dict["hook_after_res1"]                # [bsz, L, n_tok, d_e]
    after_norm2  = hook_dict["hook_after_norm2"]               # [bsz, L, n_tok, d_e]
    selected     = hook_dict["hook_selected_experts"]          # [bsz, L, n_tok, top_k]
    weighted_out = hook_dict["hook_expert_weighted_outputs"]   # [bsz, L, n_tok, top_k, d_e]

    bsz, _, n_tok, _ = after_res1.shape
    bt = bsz * n_tok

    # Reshape to [bt, ...].
    after_res1_r  = after_res1.float().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, D_E)
    after_norm2_r = after_norm2.float().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, D_E)
    sel_r         = selected.long().permute(0, 2, 1, 3).reshape(bt, N_LAYERS, TOP_K)
    omega         = (weighted_out.float()
                     .permute(0, 2, 1, 3, 4)
                     .reshape(bt, N_LAYERS, TOP_K, D_E))   # [bt, S, k, d_e]

    rms_inv = torch.rsqrt(after_res1_r.pow(2).mean(dim=-1) + EPS)  # [bt, L_recv]

    # Original assignment scores at every receiver layer.
    orig_score = torch.einsum("lnd,bld->bln", G, after_norm2_r)  # [bt, L, N_EXPERTS]

    # Original ranking + top-K.
    orig_sorted = torch.argsort(orig_score, dim=-1, descending=True)
    orig_rank_of = torch.empty_like(orig_sorted)
    orig_rank_of.scatter_(-1, orig_sorted,
                          torch.arange(N_EXPERTS, device=device).expand_as(orig_sorted))
    orig_top_k = orig_sorted[:, :, :TOP_K]                       # [bt, L, top_k]
    orig_ranks_at_topk = torch.gather(orig_rank_of, -1, orig_top_k)  # [bt, L, top_k] = 0..top_k-1

    flat_attn = attention_mask.bool().reshape(bt)
    flat_tmasks = {tname: (m.reshape(bt) & flat_attn) for tname, m in tmasks.items()}

    flat_ids = input_ids.reshape(bt)

    # ---- For each ablation target, compute Δ-score, AARV, accumulate ----
    for label, cname, members in all_targets:
        cdef = CIRCUITS[cname]
        has_trigger = cdef["trigger_fn"] is not None
        do_per_token = "per_token_sum" in acc[label]
        if has_trigger:
            trig_mask = flat_tmasks[cdef["trigger_fn"]]
            ctrl_mask = (~flat_tmasks[cdef["trigger_fn"]]) & flat_attn

        delta_score = torch.zeros((bt, N_LAYERS, N_EXPERTS), dtype=torch.float32, device=device)
        for c_t, j_t in members:
            sel_at_c = sel_r[:, c_t, :]                          # [bt, top_k]
            slot_mask = (sel_at_c == j_t).float()                # [bt, top_k]
            token_active = slot_mask.sum(dim=-1) > 0.5            # [bt]
            if not token_active.any():
                continue

            # Weighted output for the slot where j_t was selected: r * e_out.
            omega_at_c = omega[:, c_t, :, :]                      # [bt, top_k, d_e]
            omega_target = (omega_at_c * slot_mask.unsqueeze(-1)).sum(dim=1)  # [bt, d_e]

            for l in range(c_t + 1, N_LAYERS):
                ln_bar = (omega_target
                          * gamma[l].view(1, D_E)
                          * rms_inv[:, l].view(bt, 1))             # [bt, d_e]
                d_l = torch.einsum("nd,bd->bn", G[l], ln_bar)     # [bt, N_EXPERTS]
                delta_score[:, l, :] += d_l * token_active.float().unsqueeze(-1)
            del omega_target

        # Perturbed score and rank.
        pert_score = orig_score - delta_score
        pert_sorted = torch.argsort(pert_score, dim=-1, descending=True)
        pert_rank_of = torch.empty_like(pert_sorted)
        pert_rank_of.scatter_(-1, pert_sorted,
                              torch.arange(N_EXPERTS, device=device).expand_as(pert_sorted))

        pert_ranks_at_topk = torch.gather(pert_rank_of, -1, orig_top_k)   # [bt, L, top_k]
        rank_shift = (pert_ranks_at_topk.float() - orig_ranks_at_topk.float()).abs()
        aarv_il = rank_shift.mean(dim=-1)                          # [bt, L]

        # Accumulate per layer.
        # Only meaningful for receiver layers below the highest sender layer.
        min_sender = min(c for c, _ in members)
        for l in range(min_sender + 1, N_LAYERS):
            acc[label]["all_sum"][l]   += aarv_il[flat_attn, l].double().sum()
            acc[label]["all_count"][l] += flat_attn.sum()
            if has_trigger:
                acc[label]["trigger_sum"][l]   += aarv_il[trig_mask, l].double().sum()
                acc[label]["trigger_count"][l] += trig_mask.sum()
                acc[label]["control_sum"][l]   += aarv_il[ctrl_mask, l].double().sum()
                acc[label]["control_count"][l] += ctrl_mask.sum()

        if do_per_token:
            # Mean downstream AARV per token, indexed by token id.
            ds_mean = aarv_il[:, min_sender + 1:].mean(dim=-1)
            ds_mean = torch.where(flat_attn, ds_mean, torch.zeros_like(ds_mean))
            acc[label]["per_token_sum"].index_add_(0, flat_ids, ds_mean.double())
            acc[label]["per_token_count"].index_add_(0, flat_ids, flat_attn.long())

        del delta_score, pert_score, pert_sorted, pert_rank_of, pert_ranks_at_topk, rank_shift, aarv_il

    del hook_dict, after_res1, after_norm2, selected, weighted_out
    del after_res1_r, after_norm2_r, sel_r, omega, rms_inv, orig_score, orig_sorted
    del orig_rank_of, orig_top_k, orig_ranks_at_topk
    torch.cuda.empty_cache()

    bnum = B // BSZ + 1
    if bnum == 1 or bnum % 10 == 0 or bnum == n_batches:
        elapsed = time.time() - t_start
        rate = (bnum * BSZ) / elapsed if elapsed > 0 else 0.0
        eta = (N_PROMPTS - bnum * BSZ) / rate if rate > 0 else 0.0
        print(f"  batch {bnum:3d}/{n_batches}  elapsed={elapsed:.1f}s  "
              f"rate={rate:.1f} prompts/s  ETA={eta:.0f}s", flush=True)

print(f"\nDone in {time.time() - t_start:.1f}s.\n", flush=True)


# ---- Finalize ----
def finalize(info):
    aarv_a = (info["all_sum"] / info["all_count"].clamp(min=1).double()).cpu().numpy()
    aarv_a[(info["all_count"] == 0).cpu().numpy()] = 0.0
    out = {
        "aarv_all_per_layer":   aarv_a.tolist(),
        "all_count_per_layer":  info["all_count"].cpu().tolist(),
    }
    if "trigger_sum" in info:
        aarv_t = (info["trigger_sum"] / info["trigger_count"].clamp(min=1).double()).cpu().numpy()
        aarv_c = (info["control_sum"] / info["control_count"].clamp(min=1).double()).cpu().numpy()
        aarv_t[(info["trigger_count"] == 0).cpu().numpy()] = 0.0
        aarv_c[(info["control_count"] == 0).cpu().numpy()] = 0.0
        out["aarv_trigger_per_layer"]    = aarv_t.tolist()
        out["aarv_control_per_layer"]    = aarv_c.tolist()
        out["trigger_count_per_layer"]   = info["trigger_count"].cpu().tolist()
        out["control_count_per_layer"]   = info["control_count"].cpu().tolist()
    if "per_token_sum" in info:
        # Top-K tokens by mean induced AARV, restricted to tokens seen >= min count.
        s = info["per_token_sum"].cpu().numpy()
        n = info["per_token_count"].cpu().numpy()
        elig = n >= args.min_token_count
        per_tok_mean = np.full_like(s, fill_value=-np.inf, dtype=np.float64)
        per_tok_mean[elig] = s[elig] / np.maximum(n[elig], 1)
        # Top-K by induced AARV.
        top_idx = np.argsort(-per_tok_mean)[:args.top_tokens]
        out["top_tokens_by_aarv"] = [
            {
                "token_id": int(i),
                "token_text": tokenizer.decode([int(i)]),
                "mean_aarv": float(per_tok_mean[i]) if np.isfinite(per_tok_mean[i]) else 0.0,
                "count": int(n[i]),
            }
            for i in top_idx if np.isfinite(per_tok_mean[i])
        ]
    return out


results = {label: finalize(info) for label, info in acc.items()}

# Save JSON.
out_path = os.path.join(output_dir, f"ablation_{args.dataset}.json")
with open(out_path, "w") as f:
    json.dump({
        "results": results,
        "circuits": {cname: {"members": cdef["members"], "trigger_fn": cdef["trigger_fn"]}
                     for cname, cdef in CIRCUITS.items()},
        "random_circuits": {cname: rcs for cname, rcs in random_circuits.items()},
        "n_prompts": N_PROMPTS,
        "n_random_trials": args.n_random_trials,
        "dataset": args.dataset,
        "model": MODEL_ID,
    }, f, indent=2)
print(f"Saved {out_path}")


# ---- Pretty-print summary ----
def downstream_layers(members):
    return list(range(min(c for c, _ in members) + 1, N_LAYERS))


def mean_aarv(per_layer, layers):
    return float(np.mean([per_layer[l] for l in layers]))


print("\n==================== Ablation summary ====================\n")
for cname, cdef in CIRCUITS.items():
    members = cdef["members"]
    members_str = ", ".join(f"M{c}E{j}" for c, j in members)
    layers = downstream_layers(members)
    has_trigger = cdef["trigger_fn"] is not None

    real = results[f"real_{cname}"]
    real_a = mean_aarv(real["aarv_all_per_layer"], layers)
    rand_a = [mean_aarv(results[f"random_{cname}_t{i}"]["aarv_all_per_layer"], layers)
              for i in range(args.n_random_trials)]

    print(f"[{cname}]   V_C = {{{members_str}}}   trigger = {cdef['trigger_fn']!r}")

    if has_trigger:
        real_t = mean_aarv(real["aarv_trigger_per_layer"], layers)
        real_c = mean_aarv(real["aarv_control_per_layer"], layers)
        rand_t = [mean_aarv(results[f"random_{cname}_t{i}"]["aarv_trigger_per_layer"], layers)
                  for i in range(args.n_random_trials)]
        rand_c = [mean_aarv(results[f"random_{cname}_t{i}"]["aarv_control_per_layer"], layers)
                  for i in range(args.n_random_trials)]

        print(f"  Per-layer AARV with selectivity & specificity (downstream l > {min(c for c, _ in members)}):")
        print(f"    {'layer':>6} {'trig|real':>10} {'ctrl|real':>10} "
              f"{'trig|rand':>10} {'ctrl|rand':>10}   {'sel(T/C)':>9} {'spec(R/r)':>10}")
        for l in layers:
            rt = real["aarv_trigger_per_layer"][l]
            rc = real["aarv_control_per_layer"][l]
            rtr = float(np.mean([results[f"random_{cname}_t{i}"]["aarv_trigger_per_layer"][l]
                                 for i in range(args.n_random_trials)]))
            rcr = float(np.mean([results[f"random_{cname}_t{i}"]["aarv_control_per_layer"][l]
                                 for i in range(args.n_random_trials)]))
            sel_l = rt / max(rc, 1e-9)
            spec_l = rt / max(rtr, 1e-9)
            print(f"    M{l:>5d} {rt:>10.4f} {rc:>10.4f} {rtr:>10.4f} {rcr:>10.4f}   "
                  f"{sel_l:>8.2f}x {spec_l:>9.2f}x")

        print(f"\n  Aggregated across downstream layers:")
        print(f"    AARV(trigger | real circuit)   = {real_t:.4f}")
        print(f"    AARV(control | real circuit)   = {real_c:.4f}")
        print(f"    AARV(trigger | random, n={args.n_random_trials}) = {np.mean(rand_t):.4f} ± {np.std(rand_t):.4f}")
        print(f"    AARV(control | random, n={args.n_random_trials}) = {np.mean(rand_c):.4f} ± {np.std(rand_c):.4f}")

        sel = real_t / max(real_c, 1e-9)
        spec = real_t / max(np.mean(rand_t), 1e-9)
        print(f"\n  Aggregate selectivity  AARV(trigger | real) / AARV(control | real)   = {sel:.2f}x")
        print(f"  Aggregate specificity  AARV(trigger | real) / AARV(trigger | random) = {spec:.2f}x")
    else:
        print(f"  Per-layer AARV (downstream l > {min(c for c, _ in members)}):")
        print(f"    {'layer':>6} {'AARV|real':>10} {'AARV|rand':>10}   {'spec(R/r)':>10}")
        for l in layers:
            ra  = real["aarv_all_per_layer"][l]
            rar = float(np.mean([results[f"random_{cname}_t{i}"]["aarv_all_per_layer"][l]
                                 for i in range(args.n_random_trials)]))
            spec_l = ra / max(rar, 1e-9)
            print(f"    M{l:>5d} {ra:>10.4f} {rar:>10.4f}   {spec_l:>9.2f}x")

        print(f"\n  Aggregated across downstream layers:")
        print(f"    AARV(real circuit)            = {real_a:.4f}")
        print(f"    AARV(random, n={args.n_random_trials}) = {np.mean(rand_a):.4f} ± {np.std(rand_a):.4f}")
        spec = real_a / max(np.mean(rand_a), 1e-9)
        print(f"\n  Aggregate specificity  AARV(real) / AARV(random) = {spec:.2f}x")

        # Trigger-discovery diagnostic: top tokens by induced AARV.
        if "top_tokens_by_aarv" in real:
            print(f"\n  Top-{args.top_tokens} tokens by induced mean-downstream AARV "
                  f"(min count {args.min_token_count}):")
            print(f"    {'rank':>4}  {'token (decoded)':<25} {'mean AARV':>10} {'count':>8}")
            for i, tok in enumerate(real["top_tokens_by_aarv"], 1):
                text_repr = repr(tok["token_text"])[:25]
                print(f"    {i:>4}  {text_repr:<25} {tok['mean_aarv']:>10.4f} {tok['count']:>8d}")
    print()
