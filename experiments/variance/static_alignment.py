"""Tier 0 — static alignment for OLMoE-1B-7B-0924.

Mirrors the paper's variance-over-receiving-routers metric (paper §5, §7),
applied to the static factor of the score factorization. No forward passes.

The score decomposes as
    S^{l,n}_{c,j,z}(i) = (1/RMS_i) * r^{c,j}(i) * alpha^{c,j}_z(i) * A^{l,n}_{c,j,z}
where the dynamic prefactor does not depend on the receiving router n. So
routing-relevant influence (Var_n S) factorizes as
    Var_n S^{l,n}_{c,j,z}(i) = K(i, c, j, z)^2 * Var_n A^{l, n}_{c, j, z}.
Tier 0 drops the dynamic K and computes the right-hand variance from weights:
    V[c, j, z, l] = Var_n  A^{l, n}_{c, j, z}        (population variance over n)

From this single 16M-scalar tensor we derive every paper-style view:
    fig3_analog:           mean over (j, z) of V    -> [c, l]      (paper Fig 3)
    fig5_M1E9 / M4E14:     V[c, j, :, :]            -> [d_ffn, l]  (paper Fig 5)
    super_expert_ranking:  sum_{l,z} V              -> ranked (c, j)  (paper §7)
    super_neuron_ranking:  sum_l V                  -> ranked (c, j, z)

Both raw sum and count-normalized (1 / (L - c - 1)) variants are reported for
the rankings, since deeper sending layers see fewer downstream receivers.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml"), "r") as f:
    config = yaml.safe_load(f)
output_dir = os.path.join(config["result_path"], "variance")
os.makedirs(output_dir, exist_ok=True)

device = "cuda:0"
torch.set_grad_enabled(False)

from customized_models.modeling_olmoe_customized import OlmoeForCausalLM

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_LAYERS = 16
N_EXPERTS = 64
D_E = 2048
D_FFN = 1024

# Paper's super-experts in OLMoE (paper §7).
SUPER_EXPERTS = {"M1E9": (1, 9), "M4E14": (4, 14)}

print(f"Loading {MODEL_ID} ...", flush=True)
t0 = time.time()
model = OlmoeForCausalLM.from_pretrained(
    MODEL_ID, attn_implementation="eager", torch_dtype=torch.bfloat16
).to(device)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

G = torch.stack([
    model.model.layers[l].mlp.gate.weight.detach().to(device, dtype=torch.float32)
    for l in range(N_LAYERS)
])  # [L, n_experts, d_e]
gamma = torch.stack([
    model.model.layers[l].post_attention_layernorm.weight.detach().to(device, dtype=torch.float32)
    for l in range(N_LAYERS)
])  # [L, d_e]
G_tilde = G * gamma.unsqueeze(1)  # [L, n_experts, d_e] — folds gamma into the router
del G, gamma

# Atomic tensor: V[c, j, z, l] = Var_n A^{l, n}_{c, j, z}. Entries with l <= c stay 0.
V = torch.zeros((N_LAYERS, N_EXPERTS, D_FFN, N_LAYERS), dtype=torch.float32, device=device)

print("Computing V[c, j, z, l] = Var_n A^{l, n}_{c, j, z} ...", flush=True)
t0 = time.time()
for c in range(N_LAYERS):
    Wd_c = torch.stack([
        model.model.layers[c].mlp.experts[j].down_proj.weight.detach().to(device, dtype=torch.float32)
        for j in range(N_EXPERTS)
    ])  # [n_experts, d_e, d_ffn]
    for l in range(c + 1, N_LAYERS):
        A_slab = torch.einsum("nd,jdz->njz", G_tilde[l], Wd_c)  # [n_experts_recv, n_experts_send, d_ffn]
        V[c, :, :, l] = A_slab.var(dim=0, unbiased=False)
        del A_slab
    del Wd_c
    torch.cuda.empty_cache()
    print(f"  c={c:2d}: elapsed {time.time() - t0:.1f}s, "
          f"V[c].sum={V[c].sum().item():.4e} max={V[c].max().item():.4e}", flush=True)
print(f"Done in {time.time() - t0:.1f}s.\n", flush=True)

V_cpu = V.cpu()
torch.save(V_cpu, os.path.join(output_dir, "V.pt"))

# ---------------- View 1: paper Fig 3 analog -- mean per-channel variance per (c, l) ----
fig3 = V_cpu.mean(dim=(1, 2)).numpy()  # [c, l]
np.save(os.path.join(output_dir, "fig3_analog.npy"), fig3)

# ---------------- View 2: paper Fig 5 analog -- per-(neuron, recv-layer) for super-experts ----
for name, (c, j) in SUPER_EXPERTS.items():
    arr = V_cpu[c, j].numpy()  # [d_ffn, L]
    np.save(os.path.join(output_dir, f"fig5_{name}.npy"), arr)

# ---------------- View 3: super-expert ranking (paper §7 analog) -- sum_{l, z} V ----
expert_raw = V_cpu.sum(dim=(2, 3))  # [L, n_experts]
recv_count = torch.tensor([N_LAYERS - c - 1 for c in range(N_LAYERS)], dtype=torch.float32)
recv_count_safe = recv_count.clamp(min=1).view(N_LAYERS, 1)
expert_norm = expert_raw / recv_count_safe  # [L, n_experts]
expert_norm[N_LAYERS - 1] = 0.0  # c=15 has no receivers

def rank_2d(score):
    flat = score.reshape(-1)
    vals, idx = torch.sort(flat, descending=True)
    out, lookup = [], {}
    for rank, (v, i) in enumerate(zip(vals.tolist(), idx.tolist())):
        c = i // N_EXPERTS
        j = i % N_EXPERTS
        out.append({"rank": rank, "c": c, "j": j, "score": v})
        lookup[(c, j)] = rank
    return out, lookup

ranking_raw, lookup_raw = rank_2d(expert_raw)
ranking_norm, lookup_norm = rank_2d(expert_norm)
super_expert_ranking = {
    "n_experts_total": N_LAYERS * N_EXPERTS,
    "raw_sum": {
        "definition": "sum_{l > c} sum_z V[c, j, z, l]",
        "M1E9_rank": lookup_raw[SUPER_EXPERTS["M1E9"]],
        "M4E14_rank": lookup_raw[SUPER_EXPERTS["M4E14"]],
        "ranking": ranking_raw,
    },
    "count_normalized": {
        "definition": "(1/(L-c-1)) * sum_{l > c} sum_z V[c, j, z, l]",
        "M1E9_rank": lookup_norm[SUPER_EXPERTS["M1E9"]],
        "M4E14_rank": lookup_norm[SUPER_EXPERTS["M4E14"]],
        "ranking": ranking_norm,
    },
}
with open(os.path.join(output_dir, "super_expert_ranking.json"), "w") as f:
    json.dump(super_expert_ranking, f, indent=2)

# ---------------- View 4: super-neuron ranking -- sum_l V[c, j, z, l] ----
neuron_raw = V_cpu.sum(dim=3)  # [L, n_experts, d_ffn]
neuron_norm = neuron_raw / recv_count_safe.view(N_LAYERS, 1, 1)
neuron_norm[N_LAYERS - 1] = 0.0

def top_k_neurons(score, k=100):
    flat = score.reshape(-1)
    vals, idx = torch.topk(flat, k)
    out = []
    for v, i in zip(vals.tolist(), idx.tolist()):
        c = i // (N_EXPERTS * D_FFN)
        rest = i % (N_EXPERTS * D_FFN)
        j = rest // D_FFN
        z = rest % D_FFN
        in_super = None
        for name, (cc, jj) in SUPER_EXPERTS.items():
            if c == cc and j == jj:
                in_super = name
                break
        out.append({"c": c, "j": j, "z": z, "score": v, "in_super_expert": in_super})
    return out

top100_raw = top_k_neurons(neuron_raw, 100)
top100_norm = top_k_neurons(neuron_norm, 100)
super_neuron_ranking = {
    "n_total_neurons": N_LAYERS * N_EXPERTS * D_FFN,
    "raw_sum": {
        "definition": "sum_{l > c} V[c, j, z, l]",
        "top100": top100_raw,
        "n_top100_inside_super_experts": sum(1 for x in top100_raw if x["in_super_expert"]),
    },
    "count_normalized": {
        "definition": "(1/(L-c-1)) * sum_{l > c} V[c, j, z, l]",
        "top100": top100_norm,
        "n_top100_inside_super_experts": sum(1 for x in top100_norm if x["in_super_expert"]),
    },
}
with open(os.path.join(output_dir, "super_neuron_ranking.json"), "w") as f:
    json.dump(super_neuron_ranking, f, indent=2)

# ---------------- Concentration of super-neuron mass ----
def concentration(score):
    flat = score.reshape(-1)
    n = flat.numel()
    total = flat.sum().item()
    sv, _ = torch.sort(flat, descending=True)
    return {
        "total": total,
        "n": n,
        "top_100": sv[:100].sum().item() / total,
        "top_0_1pct": sv[:max(1, n // 1000)].sum().item() / total,
        "top_1pct": sv[:max(1, n // 100)].sum().item() / total,
        "top_10pct": sv[:max(1, n // 10)].sum().item() / total,
    }

concentrations = {
    "raw_sum": concentration(neuron_raw),
    "count_normalized": concentration(neuron_norm),
}
with open(os.path.join(output_dir, "concentration.json"), "w") as f:
    json.dump(concentrations, f, indent=2)

# ---------------- Stdout summary ----------------
def print_top10(rows, label):
    print(f"\n{label}:")
    for x in rows[:10]:
        tag = f"  [in {x['in_super_expert']}]" if x.get("in_super_expert") else ""
        if "z" in x:
            print(f"  c={x['c']:2d} j={x['j']:3d} z={x['z']:4d}  score={x['score']:.4e}{tag}")
        else:
            print(f"  rank={x['rank']:3d} c={x['c']:2d} j={x['j']:3d}  score={x['score']:.4e}")

print("==================== Tier 0 summary ====================\n")

print("Concentration of sum_l V (raw):")
for k, v in concentrations["raw_sum"].items():
    print(f"  {k:>11}: {v}")
print("\nConcentration of sum_l V (count-normalized):")
for k, v in concentrations["count_normalized"].items():
    print(f"  {k:>11}: {v}")

print("\n----- Paper Fig. 3 analog: mean_{j,z} V[c, j, z, l]  (rows=c, cols=l) -----")
print("  (top-left blank — l > c only)")
print("       " + "  ".join(f"l={l:2d}" for l in range(N_LAYERS)))
for c in range(N_LAYERS):
    row = "  ".join(f"{fig3[c, l]:.3f}" if l > c else " .   " for l in range(N_LAYERS))
    print(f"  c={c:2d} {row}")

print("\n----- Super-expert ranking (raw sum) -----")
print(f"  M1E9  (c=1, j=9 ): rank {super_expert_ranking['raw_sum']['M1E9_rank']:4d} / {N_LAYERS*N_EXPERTS}")
print(f"  M4E14 (c=4, j=14): rank {super_expert_ranking['raw_sum']['M4E14_rank']:4d} / {N_LAYERS*N_EXPERTS}")
print_top10([{"rank": r["rank"], "c": r["c"], "j": r["j"], "score": r["score"]} for r in ranking_raw[:10]],
            "Top-10 experts (raw)")

print("\n----- Super-expert ranking (count-normalized) -----")
print(f"  M1E9  (c=1, j=9 ): rank {super_expert_ranking['count_normalized']['M1E9_rank']:4d} / {N_LAYERS*N_EXPERTS}")
print(f"  M4E14 (c=4, j=14): rank {super_expert_ranking['count_normalized']['M4E14_rank']:4d} / {N_LAYERS*N_EXPERTS}")
print_top10([{"rank": r["rank"], "c": r["c"], "j": r["j"], "score": r["score"]} for r in ranking_norm[:10]],
            "Top-10 experts (count-normalized)")

print(f"\n----- Super-neuron ranking (raw) -- Σ_l V[c, j, z, l] -----")
print(f"  Top-100 inside paper super-experts: "
      f"{super_neuron_ranking['raw_sum']['n_top100_inside_super_experts']} / 100")
print_top10(top100_raw, "Top-10 neurons (raw)")

print(f"\n----- Super-neuron ranking (count-normalized) -----")
print(f"  Top-100 inside paper super-experts: "
      f"{super_neuron_ranking['count_normalized']['n_top100_inside_super_experts']} / 100")
print_top10(top100_norm, "Top-10 neurons (count-normalized)")

# Where do the M1E9 / M4E14 neurons fall in the global super-neuron ranking?
flat_raw = neuron_raw.reshape(-1)
order_raw = torch.argsort(flat_raw, descending=True)
rank_of = torch.empty_like(order_raw)
rank_of[order_raw] = torch.arange(order_raw.numel())
print("\n----- Top-5 neurons inside each paper super-expert (raw ranking) -----")
for name, (c, j) in SUPER_EXPERTS.items():
    scores = neuron_raw[c, j]  # [d_ffn]
    top_z = torch.topk(scores, 5)
    print(f"  {name} (c={c}, j={j}):")
    for v, z in zip(top_z.values.tolist(), top_z.indices.tolist()):
        flat_idx = c * N_EXPERTS * D_FFN + j * D_FFN + z
        gr = rank_of[flat_idx].item()
        print(f"    z={z:4d}  score={v:.4e}  global super-neuron rank={gr}")

print(f"\nArtifacts written to {output_dir}")
