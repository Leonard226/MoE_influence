"""Tier 1 — full per-neuron variance with dynamic factors.

Computes the paper-faithful per-neuron variance metric on real C4 forward passes:

    T_dyn[c, j, z, l]  =  E_i [ Var_n S^{l,n}_{c,j,z}(i) | expert (c,j) selected at i ]

Using the score factorization
    S^{l,n}_{c,j,z}(i) = (1/RMS^l_i) * r^{c,j}(i) * alpha^{c,j}_z(i) * A^{l,n}_{c,j,z}
the dynamic prefactor is independent of n, so
    Var_n S^{l,n}_{c,j,z}(i) = [(1/RMS^l_i) * r^{c,j}(i) * alpha^{c,j}_z(i)]^2 * V_static[c,j,z,l]
where V_static = Var_n A is the precomputed Tier 0 tensor.

Output (under config.yaml:result_path / variance/):
    T_dyn.pt                       full [L, n_experts, d_ffn, L] fp32 tensor
    count.pt                       per-(c, j) selection count (paper "occur_counter")
    super_expert_ranking_dyn.json  sum_{l,z} T_dyn ranked across (c, j) — paper §7 analog
    super_neuron_ranking_dyn.json  sum_l T_dyn ranked across (c, j, z) — top-100
"""
import json
import os
import sys
import time

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
output_dir = os.path.join(config["result_path"], "variance")
os.makedirs(output_dir, exist_ok=True)

device = "cuda:0"
torch.set_default_device(device)
torch.set_grad_enabled(False)

from customized_models.modeling_olmoe_customized import OlmoeForCausalLM
from transformers import AutoTokenizer
from dataset.c4_dataset import c4_dataset_helper

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_LAYERS = 16
N_EXPERTS = 64
D_FFN = 1024
TOP_K = 8
EPS = 1e-5
SUPER_EXPERTS = {"M1E9": (1, 9), "M4E14": (4, 14)}

# Paper scale: 5000 prompts × 32 tokens, batch size 50.
N_PROMPTS = 5000
BSZ = 50
MAX_TOKENS = 32

# ---- Load V_static from Tier 0 ----
v_path = os.path.join(config["result_path"], "variance", "V.pt")
print(f"Loading V_static from {v_path} ...", flush=True)
V_static = torch.load(v_path, map_location=device)  # [L, n_experts, d_ffn, L]
assert V_static.shape == (N_LAYERS, N_EXPERTS, D_FFN, N_LAYERS), V_static.shape

# ---- Load model + tokenizer ----
print(f"Loading {MODEL_ID} ...", flush=True)
t0 = time.time()
model = OlmoeForCausalLM.from_pretrained(
    MODEL_ID, attn_implementation="eager", torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

# ---- Load C4 ----
print(f"Loading C4 ({N_PROMPTS} prompts, min_words={MAX_TOKENS}) ...", flush=True)
t0 = time.time()
prompts = c4_dataset_helper(dataset_len=N_PROMPTS, seed=None, min_words=MAX_TOKENS)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

# ---- Init accumulators ----
T_dyn_accum = torch.zeros((N_LAYERS, N_EXPERTS, D_FFN, N_LAYERS), dtype=torch.float32, device=device)
count = torch.zeros((N_LAYERS, N_EXPERTS), dtype=torch.long, device=device)

n_batches = (N_PROMPTS + BSZ - 1) // BSZ
print(f"Running {n_batches} batches of {BSZ} prompts × {MAX_TOKENS} tokens ...", flush=True)
t_start = time.time()

for B in range(0, N_PROMPTS, BSZ):
    batch = prompts[B:B + BSZ]
    inputs = tokenizer(batch, return_tensors="pt", padding=False, truncation=True, max_length=MAX_TOKENS)
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)

    _, hook_dict = model(input_ids=input_ids, attention_mask=attention_mask)

    after_res1 = hook_dict["hook_after_res1"]              # [bsz, L, n_tok, d_e]
    routing_w = hook_dict["hook_routing_weights"]          # [bsz*n_tok, L, top_k]
    selected = hook_dict["hook_selected_experts"]          # [bsz, L, n_tok, top_k]
    alpha = hook_dict["hook_alpha"]                        # [bsz, L, n_tok, top_k, d_ffn]

    bsz, _, n_tok, _ = after_res1.shape
    bt = bsz * n_tok

    # rsq_inv[l, i] = 1 / RMS^l_i^2
    rms_sq = after_res1.float().pow(2).mean(dim=-1) + EPS  # [bsz, L, n_tok]
    rsq_inv = (1.0 / rms_sq).permute(1, 0, 2).reshape(N_LAYERS, bt)  # [L, bt]

    # Reshape per-c quantities: [L, bt, top_k, ...]
    r_sq_all = routing_w.float().pow(2).permute(1, 0, 2)  # [L, bt, top_k]
    alpha_sq_all = (alpha.float()
                    .pow(2)
                    .permute(1, 0, 2, 3, 4)
                    .reshape(N_LAYERS, bt, TOP_K, D_FFN))  # [L, bt, top_k, d_ffn]
    sel_all = (selected.long()
               .permute(1, 0, 2, 3)
               .reshape(N_LAYERS, bt, TOP_K))  # [L, bt, top_k]

    for c in range(N_LAYERS):
        sel_c = sel_all[c]                          # [bt, top_k]
        count[c] += torch.bincount(sel_c.flatten(), minlength=N_EXPERTS)

        if c == N_LAYERS - 1:
            continue  # no downstream receivers

        r_sq_c = r_sq_all[c]                        # [bt, top_k]
        alpha_sq_c = alpha_sq_all[c]                # [bt, top_k, d_ffn]
        # q[i, k, z] = r²[i, k] · α²[i, k, z]
        q_c = r_sq_c.unsqueeze(-1) * alpha_sq_c     # [bt, top_k, d_ffn]

        # V_for_c[i, k, z, l] = V_static[c, sel_c[i, k], z, l]
        V_for_c = V_static[c][sel_c]                # [bt, top_k, d_ffn, L]

        # K_sq[i, k, z, l] = (1/RMS^l_i)² · q_c[i, k, z]
        # rsq_inv[L, bt] -> view to [bt, 1, 1, L]
        rsq_t = rsq_inv.permute(1, 0).view(bt, 1, 1, N_LAYERS)
        contrib = (q_c.unsqueeze(-1) * V_for_c) * rsq_t  # [bt, top_k, d_ffn, L]

        # Zero out l ≤ c (Tier 0 already has V=0 there but keep explicit for clarity).
        contrib[..., :c + 1] = 0.0

        # Scatter-add into T_dyn_accum[c, j, :, :] indexed by j = sel_c[i, k].
        T_dyn_accum[c].index_add_(
            0,
            sel_c.flatten(),
            contrib.reshape(bt * TOP_K, D_FFN, N_LAYERS),
        )

        del q_c, V_for_c, contrib, rsq_t

    del hook_dict, after_res1, routing_w, selected, alpha, rsq_inv, rms_sq
    del r_sq_all, alpha_sq_all, sel_all
    torch.cuda.empty_cache()

    bnum = B // BSZ + 1
    if bnum == 1 or bnum % 10 == 0 or bnum == n_batches:
        elapsed = time.time() - t_start
        rate = (bnum * BSZ) / elapsed
        eta = (N_PROMPTS - bnum * BSZ) / rate if rate > 0 else 0.0
        print(f"  batch {bnum:3d}/{n_batches}  elapsed={elapsed:.1f}s  "
              f"rate={rate:.1f} prompts/s  ETA={eta:.0f}s", flush=True)

print(f"\nDone in {time.time() - t_start:.1f}s.\n", flush=True)

# ---- Normalize: T_dyn[c, j, z, l] = T_dyn_accum / count[c, j] ----
count_safe = count.clamp(min=1).to(torch.float32)
T_dyn = T_dyn_accum / count_safe.view(N_LAYERS, N_EXPERTS, 1, 1)
T_dyn = T_dyn.masked_fill((count == 0).view(N_LAYERS, N_EXPERTS, 1, 1), 0.0)
T_dyn_cpu = T_dyn.cpu()

torch.save(T_dyn_cpu, os.path.join(output_dir, "T_dyn.pt"))
torch.save(count.cpu(), os.path.join(output_dir, "count.pt"))

# ---- Super-expert ranking ----
expert_score = T_dyn_cpu.sum(dim=(2, 3))  # [L, n_experts]
expert_flat = expert_score.reshape(-1)
vals, idx = torch.sort(expert_flat, descending=True)
ranking = []
m1e9_rank = m4e14_rank = None
for r, (v, i) in enumerate(zip(vals.tolist(), idx.tolist())):
    c = i // N_EXPERTS
    j = i % N_EXPERTS
    ranking.append({"rank": r, "c": c, "j": j, "score": v, "count": int(count[c, j].item())})
    if (c, j) == SUPER_EXPERTS["M1E9"]:
        m1e9_rank = r
    if (c, j) == SUPER_EXPERTS["M4E14"]:
        m4e14_rank = r

with open(os.path.join(output_dir, "super_expert_ranking_dyn.json"), "w") as f:
    json.dump({
        "n_experts_total": N_LAYERS * N_EXPERTS,
        "M1E9_rank": m1e9_rank,
        "M4E14_rank": m4e14_rank,
        "ranking": ranking,
    }, f, indent=2)

# ---- Super-neuron ranking ----
neuron_score = T_dyn_cpu.sum(dim=3)  # [L, n_experts, d_ffn]
flat = neuron_score.reshape(-1)
top_vals, top_idx = torch.topk(flat, 100)
top100 = []
n_in_super = 0
for v, i in zip(top_vals.tolist(), top_idx.tolist()):
    c = i // (N_EXPERTS * D_FFN)
    rest = i % (N_EXPERTS * D_FFN)
    j = rest // D_FFN
    z = rest % D_FFN
    in_super = next((name for name, (cc, jj) in SUPER_EXPERTS.items() if c == cc and j == jj), None)
    if in_super is not None:
        n_in_super += 1
    top100.append({"c": c, "j": j, "z": z, "score": v, "in_super_expert": in_super})

with open(os.path.join(output_dir, "super_neuron_ranking_dyn.json"), "w") as f:
    json.dump({
        "n_total_neurons": N_LAYERS * N_EXPERTS * D_FFN,
        "n_top100_inside_super_experts": n_in_super,
        "top100": top100,
    }, f, indent=2)

# ---- Top-5 inside each paper super-expert ----
order = torch.argsort(flat, descending=True)
rank_of = torch.empty_like(order)
rank_of[order] = torch.arange(order.numel(), device=order.device)

# ---- Stdout summary ----
print("==================== Tier 1 summary ====================\n")
print(f"Tokens processed: {N_PROMPTS} prompts × {MAX_TOKENS} tokens (batch={BSZ})")
print(f"\n--- Super-expert ranking (dynamic) ---")
print(f"  M1E9  (c=1, j=9 ): rank {m1e9_rank} / {N_LAYERS * N_EXPERTS}  "
      f"(Tier 0 raw was 100)")
print(f"  M4E14 (c=4, j=14): rank {m4e14_rank} / {N_LAYERS * N_EXPERTS}  "
      f"(Tier 0 raw was 139)")
print("\nTop-10 experts:")
for r in ranking[:10]:
    cc, jj = r["c"], r["j"]
    tag = ""
    for name, (sc, sj) in SUPER_EXPERTS.items():
        if (cc, jj) == (sc, sj):
            tag = f"  <-- {name}"
    print(f"  rank={r['rank']:3d}  c={cc:2d} j={jj:3d}  score={r['score']:.4e}  count={r['count']}{tag}")

print(f"\n--- Super-neuron ranking (dynamic) ---")
print(f"Top-100 inside paper super-experts: {n_in_super} / 100  (Tier 0 raw was 4/100)")
print("\nTop-10 neurons:")
for n in top100[:10]:
    tag = f"  [in {n['in_super_expert']}]" if n["in_super_expert"] else ""
    print(f"  c={n['c']:2d} j={n['j']:3d} z={n['z']:4d}  score={n['score']:.4e}{tag}")

print("\n--- Top-5 neurons inside each paper super-expert (dynamic ranking) ---")
for name, (c, j) in SUPER_EXPERTS.items():
    scores = neuron_score[c, j]  # [d_ffn]
    top_z = torch.topk(scores, 5)
    print(f"  {name} (c={c}, j={j}, count={int(count[c, j].item())}):")
    for v, z in zip(top_z.values.tolist(), top_z.indices.tolist()):
        flat_idx = c * N_EXPERTS * D_FFN + j * D_FFN + z
        gr = int(rank_of[flat_idx].item())
        print(f"    z={z:4d}  score={v:.4e}  global super-neuron rank={gr}")

print(f"\nArtifacts written to {output_dir}")
