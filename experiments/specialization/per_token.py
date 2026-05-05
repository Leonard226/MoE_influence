"""Per-token activation accumulator for named neurons (unconditional).

For each target neuron (c, j, z), accumulate (sum |alpha|, count) per
input-token id over the C4 corpus. Activation is computed UNCONDITIONALLY:
we evaluate the named expert's SwiGLU intermediate
    |sigma(W_g_{z,:} m_in) * (W_u_{z,:} m_in)|
at every token position regardless of whether the routing gate selected
expert (c, j) at that position. This removes the selection-bias of the
earlier conditional version and gives a clean feature-detector reading
("which tokens make z fire if the expert were forced to evaluate them").

Output (under config.yaml:result_path / specialization/):
    per_token_activation.pt   dict: name -> {c, j, z, sum[V], count[V]}
"""
import os
import sys
import time

import torch
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
output_dir = os.path.join(config["result_path"], "specialization")
os.makedirs(output_dir, exist_ok=True)

device = "cuda:0"
torch.set_default_device(device)
torch.set_grad_enabled(False)

from customized_models.modeling_olmoe_customized import OlmoeForCausalLM
from transformers import AutoTokenizer
from dataset.c4_dataset import c4_dataset_helper

MODEL_ID = "allenai/OLMoE-1B-7B-0924"
N_PROMPTS = 25000
BSZ = 50
MAX_TOKENS = 32

# (name, c, j, z)
TARGETS = [
    ("M1E9_z915",    1,  9, 915),
    ("M2E30_z742",   2, 30, 742),
    ("M4E14_z391",   4, 14, 391),
    ("M4E14_z336",   4, 14, 336),
    ("M4E14_z956",   4, 14, 956),
    ("M14E60_z357", 14, 60, 357),
]

print(f"Loading {MODEL_ID} ...", flush=True)
t0 = time.time()
model = OlmoeForCausalLM.from_pretrained(
    MODEL_ID, attn_implementation="eager", torch_dtype=torch.bfloat16
).to(device).eval()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
VOCAB_SIZE = len(tokenizer)
print(f"  loaded in {time.time() - t0:.1f}s, vocab={VOCAB_SIZE}", flush=True)

# Precompute W_g_z and W_u_z for each target (sliced from the full expert weights).
W_g_named = {}
W_u_named = {}
for name, c, j, z in TARGETS:
    expert = model.model.layers[c].mlp.experts[j]
    W_g_named[name] = expert.gate_proj.weight[z].detach().to(torch.float32).to(device)  # [d_e]
    W_u_named[name] = expert.up_proj.weight[z].detach().to(torch.float32).to(device)    # [d_e]

print(f"Loading C4 ({N_PROMPTS} prompts) ...", flush=True)
t0 = time.time()
prompts = c4_dataset_helper(dataset_len=N_PROMPTS, seed=None, min_words=MAX_TOKENS)
print(f"  loaded in {time.time() - t0:.1f}s", flush=True)

# Per-target accumulators on GPU.
sums   = {name: torch.zeros(VOCAB_SIZE, dtype=torch.float64, device=device) for name, *_ in TARGETS}
counts = {name: torch.zeros(VOCAB_SIZE, dtype=torch.int64,   device=device) for name, *_ in TARGETS}

silu = torch.nn.functional.silu

n_batches = (N_PROMPTS + BSZ - 1) // BSZ
print(f"Running {n_batches} batches ...", flush=True)
t_start = time.time()

for B in range(0, N_PROMPTS, BSZ):
    batch = prompts[B:B + BSZ]
    inputs = tokenizer(batch, return_tensors="pt", padding=False, truncation=True, max_length=MAX_TOKENS)
    input_ids = inputs["input_ids"].to(device)
    attn_mask = inputs["attention_mask"].to(device)

    _, hooks = model(input_ids=input_ids, attention_mask=attn_mask)
    after_norm2 = hooks["hook_after_norm2"]   # [bsz, L, n_tok, d_e] -- input to every MoE layer

    active = attn_mask.bool()                  # [bsz, n_tok]
    active_ids = input_ids[active].to(torch.int64)  # [n_active] (same for all targets)
    ones = torch.ones_like(active_ids)

    for name, c, j, z in TARGETS:
        m_in_c = after_norm2[:, c, :, :].float()                    # [bsz, n_tok, d_e]
        g = m_in_c @ W_g_named[name]                                 # [bsz, n_tok]
        u = m_in_c @ W_u_named[name]                                 # [bsz, n_tok]
        per_pos = (silu(g) * u).abs()                                # [bsz, n_tok]

        active_vals = per_pos[active].double()                       # [n_active]
        sums[name].scatter_add_(0, active_ids, active_vals)
        counts[name].scatter_add_(0, active_ids, ones)

    del hooks, after_norm2
    torch.cuda.empty_cache()

    bnum = B // BSZ + 1
    if bnum == 1 or bnum % 50 == 0 or bnum == n_batches:
        elapsed = time.time() - t_start
        rate = (bnum * BSZ) / elapsed if elapsed > 0 else 0.0
        eta = (N_PROMPTS - bnum * BSZ) / rate if rate > 0 else 0.0
        print(f"  batch {bnum:4d}/{n_batches}  elapsed={elapsed:.1f}s  "
              f"rate={rate:.1f} prompts/s  ETA={eta:.0f}s", flush=True)

print(f"\nDone in {time.time() - t_start:.1f}s.\n", flush=True)

output = {
    name: {
        "c": c, "j": j, "z": z,
        "sum":   sums[name].cpu(),
        "count": counts[name].cpu(),
    }
    for name, c, j, z in TARGETS
}
torch.save(output, os.path.join(output_dir, "per_token_activation.pt"))
print(f"Saved {os.path.join(output_dir, 'per_token_activation.pt')}")

# Quick stdout sanity check.
print("\nTop-10 trigger tokens by mean |a| (min count = 20):")
for name, c, j, z in TARGETS:
    s = sums[name].cpu().numpy()
    cnt = counts[name].cpu().numpy()
    valid = cnt >= 20
    mean = (s / cnt.clip(min=1))
    mean[~valid] = 0.0
    top_idx = mean.argsort()[::-1][:10]
    print(f"  {name} (c={c}, j={j}, z={z}):")
    for t in top_idx:
        decoded = tokenizer.decode([int(t)]).replace("\n", "\\n")
        print(f"    {repr(decoded):>20}  mean={mean[t]:.3f}  n={int(cnt[t])}")
