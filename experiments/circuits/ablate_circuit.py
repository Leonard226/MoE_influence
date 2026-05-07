"""Joint ablation of candidate circuits with selectivity test.

For each candidate circuit V_C, jointly zero the score contributions of all members
(via the score-decomposition trick — no actual model surgery) and measure the
downstream AARV separately on trigger tokens vs control tokens. Compare to ablating
size-matched random expert sets to control for general expert importance.

Two circuits tested:
    C1 = {M1E9, M4E14}, hypothesized determiner-detection chain.
    C2 = {M2E30},       hypothesized capitalization-inhibition source.

Each circuit defines a trigger predicate over token positions. The trigger is
defined on the *current* token, not the preceding one — the analytical score-
subtraction ablation captures per-position effects (cross-position attention
contributions are not in the score decomposition), so the relevant trigger is
the position where the circuit's detector neurons fire, which is the position
of the trigger token itself.

    determiner:    current token's stripped/lowered text in {"a", "the"}
    capitalization: current token (stripped) starts with an uppercase letter

For each (target_set, dataset) pair, we report:
    AARV(trigger) — mean rank shift on trigger-class tokens.
    AARV(control) — mean rank shift on non-trigger tokens.
    Selectivity   = AARV(trigger) / AARV(control).
    Specificity   = AARV(trigger; real circuit) / AARV(trigger; random circuit).

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
CIRCUITS = {
    "C1_determiner": {
        "members": [(1, 9), (4, 14)],            # M1E9, M4E14
        "trigger_fn": "determiner",
        "description": "M1E9 → M4E14 chain hypothesized to detect/relay determiners",
    },
    "C2_capitalization": {
        "members": [(2, 30)],                     # M2E30
        "trigger_fn": "capitalization",
        "description": "M2E30 hypothesized to inhibit late-layer experts on capitalized openers",
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
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

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
def make_accum():
    return {
        "trigger_sum":   torch.zeros(N_LAYERS, dtype=torch.float64, device=device),
        "trigger_count": torch.zeros(N_LAYERS, dtype=torch.long, device=device),
        "control_sum":   torch.zeros(N_LAYERS, dtype=torch.float64, device=device),
        "control_count": torch.zeros(N_LAYERS, dtype=torch.long, device=device),
    }


acc = {label: make_accum() for label, _, _ in all_targets}

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
    for tname in {cdef["trigger_fn"] for cdef in CIRCUITS.values()}:
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

    # ---- For each ablation target, compute Δ-score, AARV, accumulate ----
    for label, cname, members in all_targets:
        tname = CIRCUITS[cname]["trigger_fn"]
        trig_mask = flat_tmasks[tname]
        ctrl_mask = (~flat_tmasks[tname]) & flat_attn

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

        # Accumulate per layer, split by token class.
        # Only meaningful for receiver layers below the highest sender layer.
        min_sender = min(c for c, _ in members)
        for l in range(min_sender + 1, N_LAYERS):
            acc[label]["trigger_sum"][l]   += aarv_il[trig_mask, l].double().sum()
            acc[label]["trigger_count"][l] += trig_mask.sum()
            acc[label]["control_sum"][l]   += aarv_il[ctrl_mask, l].double().sum()
            acc[label]["control_count"][l] += ctrl_mask.sum()

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
    aarv_t = (info["trigger_sum"] / info["trigger_count"].clamp(min=1).double()).cpu().numpy()
    aarv_c = (info["control_sum"] / info["control_count"].clamp(min=1).double()).cpu().numpy()
    aarv_t[(info["trigger_count"] == 0).cpu().numpy()] = 0.0
    aarv_c[(info["control_count"] == 0).cpu().numpy()] = 0.0
    return {
        "aarv_trigger_per_layer": aarv_t.tolist(),
        "aarv_control_per_layer": aarv_c.tolist(),
        "trigger_count_per_layer": info["trigger_count"].cpu().tolist(),
        "control_count_per_layer": info["control_count"].cpu().tolist(),
    }


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

    real = results[f"real_{cname}"]
    real_t = mean_aarv(real["aarv_trigger_per_layer"], layers)
    real_c = mean_aarv(real["aarv_control_per_layer"], layers)

    rand_t = [mean_aarv(results[f"random_{cname}_t{i}"]["aarv_trigger_per_layer"], layers)
              for i in range(args.n_random_trials)]
    rand_c = [mean_aarv(results[f"random_{cname}_t{i}"]["aarv_control_per_layer"], layers)
              for i in range(args.n_random_trials)]

    print(f"[{cname}]   V_C = {{{members_str}}}   trigger = {cdef['trigger_fn']!r}")
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
    print()
