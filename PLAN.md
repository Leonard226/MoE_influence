# Project plan — per-neuron extension of MoE routing decomposition

> **For future sessions / collaborators:** start here, then read [`per_neuron_derivation.md`](per_neuron_derivation.md) for the math.

## 1. Goal

Extend the per-expert routing decomposition of Li et al. (ICLR 2026, *Understanding Cross-Layer Contributions to Mixture-of-Experts Routing in LLMs*) **down to the per-neuron level** to test whether a small number of neurons inside the paper's "super experts" disproportionately drive routing — i.e., whether **super-neurons** exist as the actual atomic unit of the super-expert phenomenon (paper §7).

## 2. Math summary

The full derivation is in [`per_neuron_derivation.md`](per_neuron_derivation.md). The key result is that the contribution of a sending MoE layer `c` to the routing score of expert `(l, n)` decomposes into per-neuron sub-scores:

```
S^{l,n}_{(c,j,z)}  =  (1/RMS) · r^{c,j} · α^{c,j}_z · A^{l,n}_{c,j,z}
                       ──┬──     ──┬──     ──┬──         ────┬─────
                       norm      gate     neuron        router-neuron
                                          activation    alignment
```

with

```
α^{c,j}_z(m^c_in)  =  σ(W_g_(z,:)^{c,j} · m^c_in) · (W_u_(z,:)^{c,j} · m^c_in)        (scalar, SwiGLU)
A^{l,n}_{c,j,z}    =  g^{l,n} · (W_d_(:,z)^{c,j} ⊙ γ^l)                              (scalar, weights only)
```

The cleanest property of this factorization: **`A` is purely structural** — computable from trained weights alone, no forward pass, no data. The other three factors are dynamic.

### Notational note

The original paper's Eq. 11 has a typo — the upper bound of the sum should be `d_ffn` (FFN intermediate dimension), not `d_e` (embedding dim). **This has been reported to the first author (Wengang Li) via Slack — do not re-flag.** We use `d_ffn` consistently throughout this project.

A second, minor inconsistency in the paper: Eq. 8's sums are 1-indexed (`Σ_{c=1}^{l}`, `Σ_{c=1}^{l-1}`) but §5's prose / Fig. 3 labels are 0-indexed (`A0`, `M0`, `M1`, etc.). Substantively harmless — just be aware when mapping between equations and figures.

### Conventions adopted in this project

| Symbol | Meaning |
|---|---|
| `d_e` | embedding / residual stream dimension (e.g. 2048 for OLMoE) |
| `d_ffn` | FFN intermediate (per-expert hidden) dimension (e.g. 1024 for OLMoE) |
| `\cdot` | reserved for **dot products** and **scalar multiplications** only |
| `\odot` | Hadamard (elementwise) product — used for `c ⊙ γ^l` in `LN-bar` |
| `LN-bar` | the **approximate** layer-norm with shared RMS denominator (paper §4.1) |
| **norm** factor | `1/RMS(x_in + a_out)` — token-global, shared across all `(c, j, z)` |
| **gate** factor | `r^{c,j}` — softmax routing weight at sending layer `c` |
| **neuron** factor | `α^{c,j}_z = σ(W_g_(z,:) · m_in) · (W_u_(z,:) · m_in)` — SwiGLU activation |
| **alignment** factor | `A^{l,n}_{c,j,z} = g^{l,n} · (W_d_(:,z)^{c,j} ⊙ γ^l)` — purely structural, weights only |

The four-factor decomposition `S^{l,n}_{(c,j,z)} = norm · gate · neuron · alignment` is the canonical form. There is also a parallel LaTeX writeup of the derivation in the user's Overleaf project (not in this repo) — but [`per_neuron_derivation.md`](per_neuron_derivation.md) is the authoritative version for cluster work.

## 3. Codebase map (the parts that matter)

Repository: `kamlams/routing_decision` (archived upstream, forked to working remote).

### Live code

| Path | Role |
|---|---|
| [`customized_models/modeling_olmoe_customized.py`](customized_models/modeling_olmoe_customized.py) | Instrumented OLMoE with output hooks. Pattern repeats for DeepSeek / Qwen / Mixtral. |
| [`tools/analyze.py`](tools/analyze.py) | All paper experiments. ~16 functions, 122 KB. **Live, extend here.** |
| [`test/test_olmoe.py`](test/test_olmoe.py) | Entry point: loads model, calls analysis functions. |
| `dataset/` | C4 + IOI loaders. |

### Frozen / not needed

- `modified_models_backup/` — archived, do not touch.
- `tools/{batch,single,verbose,misc,plot}.py` — README states "not necessary for paper experiments". Treat as reference only.

### Already implemented (reuse as-is)

| Functionality | Location | Notes |
|---|---|---|
| Eq. 8 score decomposition (paper) | [`analyze.py:167`](tools/analyze.py#L167) `rmsnorm_breakdown_batch` | Implements the shared-RMS-denominator `LN-bar` trick. Modes `TAM`, `H`, `H_simplified`, `H_agnostic`, `E`. Operates per-batch. |
| Variance / APS / ANS heatmaps (paper Fig. 3) | [`analyze.py:606`](tools/analyze.py#L606) `decompose_TAM_tril` | The main per-layer-pair variance accumulator. |
| Per-expert variance (paper Fig. 5) | [`analyze.py:1124`](tools/analyze.py#L1124) `decompose_E` | Uses `scatter_add_` to map variance to actual firing expert IDs. |
| AARV ablation (paper Fig. 6, Eq. 13) | [`analyze.py:1486`](tools/analyze.py#L1486) `AARV_expert_olmoe` | Ablates at the **score level**: subtracts a per-expert decomposed score and measures rank shift. Subtraction line is [`analyze.py:1622`](tools/analyze.py#L1622). |

### Hooks already exposed by `customized_models`

In `OlmoeSparseMoeBlock.forward` (and analogous classes for other models):

| Hook | Shape | What it gives us |
|---|---|---|
| `hook_selected_experts` | `[batch·tokens, top_k]` | top-K expert indices |
| `hook_routing_weights` | `[batch·tokens, top_k]` | post-softmax routing weights `r^{c,j}` ✓ |
| `hook_expert_weighted_outputs` | `[batch·tokens, top_k, d_e]` | `r^{c,j} · e^{c,j}_out` (full expert output, not per-neuron) |
| `router_logits` | `[batch·tokens, n_experts]` | pre-softmax gate logits |

In `OlmoeModel.forward`: `hook_layer_input`, `hook_attn_output`, `hook_after_res1`, `hook_after_norm2`, `hook_mlp_output`, `hook_layer_output`, plus attention `hook_q`, `hook_k`, `hook_v`, `hook_attn_weights`.

### What's missing for the per-neuron extension

1. **Per-neuron activations `α^{c,j}_z`** — not currently captured. Need a new hook in `OlmoeMLP.forward` ([`modeling_olmoe_customized.py:231`](customized_models/modeling_olmoe_customized.py#L231)) to expose `act_fn(gate_proj(x)) * up_proj(x)` (the pre-down-projection intermediate). Required for **Tier 1 onward only**.

### Verified weight access paths (OLMoE)

```python
# Router weights G^l   shape: [n_experts, d_e]
model.model.layers[l].mlp.gate.weight

# RMSNorm γ^l   shape: [d_e]   (the LN scale at the receiver MoE)
model.model.layers[l].post_attention_layernorm.weight

# Expert down-projection W_d^{c,j}   shape: [d_e, d_ffn]
model.model.layers[c].mlp.experts[j].down_proj.weight

# Expert gate / up projections   shape: [d_ffn, d_e] each
model.model.layers[c].mlp.experts[j].gate_proj.weight
model.model.layers[c].mlp.experts[j].up_proj.weight
```

For OLMoE-1B-7B-0924: `n_layers=16`, `n_experts=64`, `top_k=8`, `d_e=2048`, `d_ffn=1024`.

### Model dimensions (all four models tested in the paper)

| Model | HF id | Layers | Experts/layer | Top-K | `d_e` | `d_ffn` (per-expert) |
|---|---|---|---|---|---|---|
| OLMoE-1B-7B | `allenai/OLMoE-1B-7B-0924` | 16 | 64 | 8 | 2048 | 1024 |
| DeepSeek-V2-Lite | `deepseek-ai/DeepSeek-V2-Lite` | 27 (layer 0 is dense FFN, not MoE) | 64 routed + 2 shared | 6 | 2048 | 1408 |
| Qwen3-30B-A3B | `Qwen/Qwen3-30B-A3B` | 48 | 128 | 8 | 2048 | 768 |
| Mixtral-8x7B | `mistralai/Mixtral-8x7B-v0.1` | 32 | 8 | 2 | 4096 | 14336 |

We are **only using OLMoE for Tier 0–2**. Other models are deferred until OLMoE results are conclusive.

### Datasets

[`dataset/c4_dataset.py`](dataset/c4_dataset.py) — `c4_dataset_helper(dataset_len, seed, min_words)`. Standard text. Used for paper §5 / §7 results. **Required for Tier 1+.** Not used in Tier 0.

[`dataset/ioi_dataset.py`](dataset/ioi_dataset.py) — Indirect Object Identification task (paper §6.3). Out of scope for the super-neuron hunt.

[`dataset/math_dataset.py`](dataset/math_dataset.py) — OpenR1-Math subset, used in paper Appendix J for generalization checks. Out of scope unless we want a robustness check later.

## 3.5 Environment & dependencies

### Python packages

The codebase imports (verified by reading [`tools/analyze.py:1-15`](tools/analyze.py#L1)):

```
torch, transformers, einops, numpy, matplotlib, seaborn, plotly,
tqdm, nltk, scikit-learn (for sklearn.manifold.TSNE), scipy, pyyaml
```

`nltk` needs first-time downloads (already commented in analyze.py:18-19):
```python
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')
```

### For reading the paper

If you need to render the PDF (`ICLR26_MoEs.pdf`) for math/figures (recommended — pypdf mangles formulas), install:

```
pip install pypdf
brew install poppler           # macOS — gives pdftoppm, pdfinfo
apt-get install poppler-utils  # Linux
```

Then render: `pdftoppm -r 150 ICLR26_MoEs.pdf page` produces `page-01.ppm`, `page-02.ppm`, ... which the `Read` tool can ingest as images.

### `config.yaml` — needs personal override

The repo's `config.yaml` has a single key `result_path` that currently points at the original author's home directory (`/home/wengang/...`). **Override it to a path on your cluster** before running anything that writes outputs. Either edit it locally (don't commit the edit) or use a separate `config.local.yaml` and pass the path explicitly.

### GPU expectations

| Tier | Device | VRAM | Notes |
|---|---|---|---|
| Tier 0 | CPU or GPU | ~15 GB RAM (fp16 weights) | No forward passes; matmuls only. |
| Tier 1 | GPU | ≥24 GB | Forward passes on C4 batches; needs to also hold per-neuron `α` capture. |
| Tier 2 | GPU | ≥24 GB | Same as Tier 1, plus per-neuron ablation loop overhead. |

Existing scripts default to `cuda:0` ([`test/test_olmoe.py:12`](test/test_olmoe.py#L12)).

## 4. Experiment plan

### Tier 0 — static alignment (start here)

**Question:** Is super-neuron concentration baked into the trained weights — visible without ever running the model — or is it a purely dynamic phenomenon that only emerges on data?

**What we compute:**

```
A[l, n, c, j, z]  =  g^{l,n} · (W_d_(:,z)^{c,j} ⊙ γ^l)        for c < l
T_static[c, j, z] =  Σ_{l>c} Σ_n  A[l, n, c, j, z]²
```

We never store `A` in full; we accumulate `T_static` (~1M scalars for OLMoE) one `(c, l)` slab at a time.

**Cost:** weights only, no forward passes, no data. Fits in CPU RAM (~15 GB for fp16 weights). Minutes of compute.

**Deliverables:**
1. Sorted log-CDF of `T_static` over all 1M neurons.
2. Top-100 neurons by `T_static` — listed by `(layer c, expert j, neuron z)`.
3. Per-expert aggregate `T_static[c, j] = Σ_z T_static[c, j, z]`, ranked.
4. Concentration ratio: fraction of total routing capacity in top 1% of neurons.

**Decision rule (acid test):**

| Outcome | Implication | Next step |
|---|---|---|
| Heavy tail in `T_static` **AND** M1E9 / M4E14 (paper's super experts) at top of expert ranking | Super-neurons real, structurally identifiable from weights | Proceed to Tier 1 |
| Heavy tail but paper's super experts NOT at top | Super-neurons exist but driven by dynamics, not weights | Proceed to Tier 1 with different framing |
| No heavy tail | Hypothesis dead | Pivot — try the pairwise-difference (`Δ`) reformulation |

### Tier 1 — per-neuron variance maps (gated on Tier 0 success)

**What:** Reproduce paper Fig. 3 / Fig. 5 at neuron resolution. For each sending neuron `(c, j, z)`, compute the variance of its scores across receiving experts in each downstream layer.

**Required additions:**
- New hook in `OlmoeMLP.forward` capturing `α^{c,j}_z` per token: `[batch·tokens, top_k, d_ffn]`.
- New function in `analyze.py` forking `decompose_E`, indexed by neuron.

**Cost:** GPU. Forward passes through the full model on 1k–5k C4 / OpenR1-Math samples, batched (paper used batch 50 × 32 tokens). ≥24 GB VRAM recommended.

### Tier 2 — per-neuron AARV (gated on Tier 1 success)

**What:** The money-shot causal experiment. Take the top candidate super-neurons from Tier 0/1, zero out their per-neuron contribution one at a time, measure top-K rank shift in receiving layers (paper's AARV metric, Eq. 13).

**Hypothesis to falsify:** *Knocking out 5–10 specific neurons inside M1E9 reproduces most of the AARV signal that knocking out the whole 1024-neuron M1E9 expert produces.*

**Required additions:** Modify the score-subtraction in `AARV_expert_olmoe` ([`analyze.py:1622`](tools/analyze.py#L1622)) to subtract per-neuron scores rather than per-expert scores.

## 5. Branch / fork status

- Original repo: `kamlams/routing_decision` (public, archived). Tracked as `upstream`.
- Working fork: `Leonard226/routing_decision` (writable). Tracked as `origin`.
- Working branch: `super-neurons` (planned).

## 6. Cold-start onboarding for a new Claude Code session

This section is **for a fresh agent picking up the work without any prior conversation context**. The user will not be repeating any previous discussion. **Do all of the following, in order, before writing or modifying a single line of code.** This is not optional — the previous agent built up substantial context (paper reading, full-codebase analysis, math derivation, experiment design, decision rules) that you do not have, and your job in onboarding is to reconstruct an equivalent level of grounding from the artifacts in this repo.

### Step 1 — Read the project context (mandatory)

- [`PLAN.md`](PLAN.md) — this file
- [`per_neuron_derivation.md`](per_neuron_derivation.md) — the full math derivation

### Step 2 — Read the paper (mandatory)

The paper is [`ICLR26_MoEs.pdf`](ICLR26_MoEs.pdf) at the repo root. It is **27 pages**; pypdf text extraction mangles formulas, so render to images for the math/figures (`pdftoppm -r 150` then `Read` the PNGs). The following sections are essential:

- **§3 (Background)** — Eq. 1–7, the MoE-Transformer notation. All variable names in this project (`g^{l,n}`, `m_in`, `e_out`, `r^{c,j}`, `γ^l`, etc.) come from here.
- **§4.1 (Decomposition of expert assignment scores)** — Eq. 8–11, the recursive decomposition. **Note Eq. 11 has a known typo: the sum bound should be `d_ffn`, not `d_e` — see §2 of this `PLAN.md`.**
- **§5 (Score distribution)** — the variance / APS / ANS analysis methodology and the per-layer findings (Fig. 3 stripes, etc.).
- **§7 (Scoring of experts)** — the super-expert finding (M1E9, M4E14 in OLMoE; M2E92, M3E82 in Qwen) and the AARV metric (Eq. 13). This is the phenomenon we are extending.

Appendices A (proofs of Prop. 1–2), H (super expert rank table), and I (M1E9/M4E14 ablation) are useful but not mandatory on first read.

### Step 3 — Verify codebase entry points still match `PLAN.md`

Files may have moved or been edited since `PLAN.md` was written. Spot-check by reading ±30 lines around each line number cited in §3 of this plan:

- [`analyze.py:167`](tools/analyze.py#L167) — `rmsnorm_breakdown_batch`
- [`analyze.py:606`](tools/analyze.py#L606) — `decompose_TAM_tril`
- [`analyze.py:1124`](tools/analyze.py#L1124) — `decompose_E`
- [`analyze.py:1486`](tools/analyze.py#L1486) — `AARV_expert_olmoe`, with the score-subtraction line at [`analyze.py:1622`](tools/analyze.py#L1622)
- [`modeling_olmoe_customized.py:621`](customized_models/modeling_olmoe_customized.py#L621) — `OlmoeSparseMoeBlock`, hooks at lines 643–657
- [`modeling_olmoe_customized.py:220`](customized_models/modeling_olmoe_customized.py#L220) — `OlmoeMLP` (where the missing `α^{c,j}_z` hook would go for Tier 1)

Also confirm the weight access paths still resolve by inspecting the OLMoE customized model:
- `model.model.layers[l].mlp.gate.weight`
- `model.model.layers[l].post_attention_layernorm.weight`
- `model.model.layers[c].mlp.experts[j].down_proj.weight`

### Step 4 — Get oriented in the broader codebase

You don't need to read every file, but skim:
- [`tools/`](tools/) — list and one-line description of each `.py` file
- [`customized_models/`](customized_models/) — note that there are parallel files for OLMoE / DeepSeek / Qwen / Mixtral / GPT-2; we're starting with OLMoE only
- [`test/test_olmoe.py`](test/test_olmoe.py) — the canonical entry-point pattern, useful as a template
- [`config.yaml`](config.yaml) — single key `result_path` that needs overriding for your environment (currently points at the original author's home directory)

### Step 5 — Check current git state

```
git log --oneline -20
git status
git branch -a
```

to see what has actually been committed on the `super-neurons` branch vs. what is still pending in this plan. **Do not assume any code from §4 has been written until verified by `git log` / file inspection.**

### Step 6 — Confirm before coding

Before writing or modifying any code, produce a short status report (5–15 lines) summarizing:

- **Current state of the branch** (per `git log` and file content): what's been committed, what hasn't.
- **Your understanding of the next concrete step** per §4 of this plan.
- **Any discrepancies** between this plan and the actual repo state — e.g. line numbers that have shifted, files that have moved, functions that have been renamed.
- **Any questions you have** about ambiguities in the plan that would benefit from user input before writing code.

Wait for user confirmation on that report before proceeding. The user will catch any drift between plan and reality faster than you will.

### What "the same level of grounding" means

By the time you start coding, you should be able to answer all of the following without re-reading anything:

- What is `S^{l,n}_{(c,j,z)}`, what are its four factors, and what does each factor depend on?
- Why is the `alignment` factor `A` special? What does "static" mean?
- Where in `analyze.py` does the existing per-expert decomposition live, and what is the einsum pattern?
- What is the AARV metric and what does it ablate (output? score? something else?)
- Why do we start with Tier 0 and not jump straight to a forward-pass experiment?
- What are M1E9 and M4E14 in OLMoE, and why do they matter for our acid test?
- What is the Eq. 11 typo in the paper? What's the corrected form?

If you cannot answer any of these from your reading, go back to the relevant source (paper section, math doc, or codebase entry point) before continuing.

## 7. Open questions / decisions deferred

- Whether to also compute a **dynamic** ranking `T_dyn[c, j, z] = E_token[Var_n(α · A)]` and compare with `T_static` — useful to see how much of the static ordering survives data. Currently in Tier 1 scope.
- Whether to do the pairwise-difference reformulation (where the norm / gate / activation factors all cancel, leaving only `Δa`) — cleaner but probably overkill before knowing Tier 0 outcome.
- Whether to extend to other models (DeepSeek, Qwen, Mixtral) — defer until OLMoE results are conclusive.
