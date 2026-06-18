# Literature Review: MoE Expert Specialization

*Synthesis for cross-model MoE routing comparison (Platonic-Representation-Hypothesis analog). Conducted 2026-06-14.*

---

## 1. What do MoE experts specialize on?

The empirical literature is unsettled and the answer depends heavily on (a) *which level* of specialization is probed and (b) *which model* is studied. Five claims emerge, partially contradictory.

### 1.1 Topic / domain specialization: mostly absent (with caveats)

The most-cited negative result is **Mixtral of Experts** (Jiang et al., arXiv:2401.04088, 2024). Section 5 of the Mixtral paper analyzes per-expert routing frequency over The Pile subsets (ArXiv, Biology, Philosophy, GitHub, DM Mathematics, etc.) and reports that "we do not observe any obvious patterns in expert assignment based on the topic." The single near-exception is *DM Mathematics*, where the distribution is "marginally different" — and the authors attribute even this to the synthetic-generation idiosyncrasies of that subset rather than to genuine "math expertise". They explicitly note that the routing is more aligned with **syntactic structure** than with semantic domain, with stronger consecutive-token repetition at higher layers.

**The Myth of Expert Specialization in MoEs** (arXiv:2604.09780, 2026) generalizes the Mixtral observation theoretically and empirically across GPT-OSS-20B, ERNIE-4.5-21B, Qwen-3-30B, Ling-mini, and Trinity-Mini-Base. Their central finding: cross-model expert overlap on identical math problems (~60%) is **indistinguishable** from the overlap of the same model on different problems. They derive Proposition 1, a tight upper bound on logit distance in terms of hidden-state distance, and conclude that "specialization is an emergent property of the representation space, not of the routing architecture itself" — routing reflects *geometry*, not *domain*. This is the strongest existing negative result and directly threatens any naive domain-histogram metric.

A partial dissent comes from **OLMoE** (Muennighoff et al., arXiv:2409.02060, 2024). Section 5.3 reports clear domain effects when domains are defined natively (k-means clustering on the unembedding matrix, *not* on external dataset labels): some OLMoE experts fire preferentially on arXiv vs GitHub content. The disagreement with Mixtral may be partly resolved by (a) OLMoE's fine-grained design (64 experts × top-8) and (b) the unembedding-clustered domain definition, which entangles "domain" with "vocabulary".

**DeepSeekMoE** (Dai et al., ACL 2024) claims fine-grained specialization but provides only *indirect* evidence: shared-expert ablation costs ~0.6 nats of Pile loss, and removing top-routed experts hurts more sharply than in GShard×1.5. There is no direct per-expert content analysis. The specialization claim is structural (fewer activated experts suffice) rather than interpretive.

### 1.2 Syntactic / POS specialization: well-supported

**ST-MoE** (Zoph et al., arXiv:2202.08906, 2022) is the seminal positive result. In their encoder–decoder T5-MoE analysis, encoder experts specialize at the *lexicon* level — punctuation, conjunctions, articles, verbs — while decoder experts show no such specialization. This asymmetry is rarely reproduced in modern decoder-only MoEs.

**Part-Of-Speech Sensitivity of Routers in Mixture of Experts Models** (Iezzi et al., COLING 2025, arXiv:2412.16971) is the most direct empirical test for this project's purposes. Across six models (dbrx-base, Mixtral-8x7B, Phi-3.5-MoE, deepseek-moe-16b-base, and two others) they probe whether POS tags can be predicted from routing paths. They report "expert specialization for specific POS categories" and "routing paths showing high predictive accuracy for POS." Importantly, they find this is stronger than Mixtral's own appendix hinted, and that it **persists across model families** — a finding that bears directly on the present project. The user's failed 5-bin POS histogram likely failed not because POS information is absent from routing but because the bins are too coarse (collapsing all functional words / all content words washes out the signal that Iezzi et al. detect with finer POS tags).

### 1.3 Token-identity specialization: the dominant signal

**OpenMoE** (Xue et al., ICML 2024, arXiv:2402.01739) coined "Context-Independent Specialization": MoE routes "based on the Token ID instead of high-level semantics," and "regardless of context, a certain token is more likely to be routed to a certain expert." They also show "Early Routing Learning" — token→expert assignments are fixed very early in pre-training and barely change. This is the strongest positive specialization signal in the literature, and it is *not* about domain or POS — it is about specific token strings.

**The Expert Strikes Back** (arXiv:2604.02178, 2026) interprets DeepSeek-MoE, Mixtral, OpenMoE, and Qwen3 at the expert level and confirms **token-level specialization** dominates: "individual experts show pronounced preferences for specific tokens or narrow feature sets" rather than tasks or domains.

This is highly relevant to the user's problem: if specialization is per-token-string, then the failed POS-bin histogram lost the signal by averaging over thousands of distinct tokens within each POS class, and the char-span fingerprint lost it because the *same token* appearing in different prompts at different char-positions looks different.

### 1.4 Position / sequence specialization: limited to encoder-level effects

ST-MoE noted decoder experts do not show position specialization. The Mixtral paper notes higher-layer consecutive-token repetition — a *local* sequence effect — but no absolute-position specialization. The Myth paper finds early layers route by token identity ignoring context, while deep layers become context-dependent. No serious paper argues that absolute token position is the primary axis of specialization.

### 1.5 Layer-dependent patterns: robust finding

Multiple papers converge here:
- **Multilingual Routing in MoE** (arXiv:2510.04694, 2026): language-specialization concentrates in *early and late* layers; middle layers are language-universal. Correlations between routing-divergence-from-English and Belebele accuracy in middle layers are r ∈ [−0.95, −0.80].
- **The Myth paper**: early layers route by token identity; deep layers are context-dependent.
- **OLMoE** Section 5: vocabulary specialization rises with depth then falls.
- **Wahib group, ICLR 2026** ("Understanding Cross-layer Contributions to MoE Routing in LLMs," OpenReview BqyPLOkxFY): MoE outputs in earlier layers contribute more to subsequent routing than attention outputs, with "MoE entanglement" causing persistent cross-layer routing correlations.

### 1.6 Architecture dependence

**A Closer Look into MoE in LLMs** (Lo et al., arXiv:2406.18219, 2024) compares Mixtral, DeepSeek-MoE, and Grok-1 directly. Models trained from scratch (DeepSeek, Grok) show near-zero expert weight cosine similarity, while Mixtral (likely upcycled from a dense model) shows 0.2–0.4. Expert output similarity *decreases* with depth then *jumps* in the final layer — a non-monotonic pattern reproducible across all three models. This is the only existing paper that does true cross-model expert comparison; we discuss it again in §3.

### 1.7 "Mostly noise"?

The Myth paper (§1.1) comes closest to a "mostly noise" verdict, but importantly does *not* argue routing is uninformative — it argues routing simply *inherits* hidden-state geometry, so any specialization claim must be re-stated as a hidden-state claim. This is a useful framing for the present project: we should design metrics that survive the Myth critique.

---

## 2. How has expert specialization been measured?

| Metric | What it measures | Strengths | Weaknesses | Cross-model comparable? |
|---|---|---|---|---|
| **Per-expert token-frequency distribution + JSD** (OLMoE §5.4; OpenMoE) | KL/JSD of expert's token distribution vs layer average → 0 = no specialization, 1 = peaked | Interpretable; bounded; works token-id level | Vocabulary-dependent — same vocab token may be a different string across models | No, unless tokenizers are aligned |
| **Routing-path POS prediction accuracy** (Iezzi et al. 2025) | Train classifier to predict POS from routing path; report accuracy above majority baseline | Theory-agnostic; directly tests POS sensitivity | Requires external POS tagger; binary "specialized or not" | Yes (POS is model-independent) |
| **Expert co-activation** (OLMoE §5.2) | Pairwise co-activation frequencies | Reveals expert team structure | Within-model only | No |
| **Sequence-level expert frequency cosine** (Myth paper) | Pool routing over a sequence; cosine between models | Naturally cross-model | Coarse — loses per-vertex specialization | Yes |
| **Top-P expert Jaccard overlap** (Myth paper) | Jaccard of high-frequency expert sets across inputs | Robust to permutations | Throws away frequency information | Yes |
| **Router Hamming similarity** (Myth paper) | Binary expert-usage overlap across tokens | Cheap | Discrete; coarse | Yes |
| **SAE-feature predictivity** (RouterInterp, Lasy et al., ICLR 2026, OpenReview 9a5i2vyMwN) | Train SAE on residual stream; find features most predictive of expert assignment; report per-expert natural-language explanation; 77% accuracy gain over baselines on gpt-oss-20b | Mechanistic; captures "disjoint union of fine-grained features" (Superposed Specialisation Hypothesis) | Requires training SAEs (expensive); not currently cross-model | Not directly — feature space is model-specific |
| **Expert Activation Norm (EAN) / pruning sensitivity** (Su et al. 2025 "Super Experts" arXiv:2507.23279; HEAPR) | Max output magnitude; pruning impact | Identifies critical experts (~0.05% of experts in Qwen3-30B drive massive activations) | Identifies a few outliers, not full specialization profile | Conceptually yes |
| **Domain specialization via unembedding k-means** (OLMoE §5.3) | Cluster on unembedding matrix → assign tokens to clusters → measure expert per-cluster firing | Model-native domains | Each model has different clusters | No |
| **Routing entropy** (many) | Entropy of per-token routing distribution | Trivial to compute | Doesn't say *what* the expert specializes on | Yes (scalar) |
| **Counterfactual / activation-patching routing analysis** ("When Are Experts Misrouted?" arXiv:2605.07260; Wahib group ICLR 2026) | Patch hidden states across layers; measure effect on routing | Causal | Within-model | Not directly |

The pattern is clear: **the metrics that work within-model (SAE features, unembedding clusters, co-activation) are not directly cross-model comparable**, and the metrics that are cross-model comparable (sequence-level pooled cosine, Jaccard) lose per-vertex information.

---

## 3. Cross-model MoE expert comparison

There is **essentially no direct prior work** doing what the user's project does (cross-family routing-DAG comparison with FGW). The closest sources:

1. **A Closer Look into MoE in LLMs** (Lo et al., 2024, arXiv:2406.18219) — the only paper I found that explicitly compares experts *across* MoE models. They compare static weight similarity and dynamic output similarity for Mixtral, DeepSeek-MoE, and Grok-1, finding that training paradigm (from-scratch vs upcycled) drives expert diversity. They do **not** compare *routing patterns* across models on shared inputs; they compare each model's internal statistics.

2. **The Myth paper** (2026) tests cross-model expert overlap on the *same* HMMT math problems across 5 models and finds ~60% overlap — but interprets this as the *null* (within-model self-overlap on different problems is similar). This is a near-direct ancestor of the present project's question.

3. **Multilingual Routing in MoE** (2026) compares routing patterns across 4 models (Qwen3-30B, Phi-3.5-MoE, GPT-OSS-20B, OLMoE) and finds "Qwen3-30B-A3B, Phi-3.5-MoE, GPT-OSS-20B, OLMoE adopt similar mechanisms" with language-agnostic middle layers. This is a *qualitative* cross-model claim; no shared metric is reported.

4. **Wahib group, ICLR 2026 (BqyPLOkxFY)** — the direct ancestor — analyzes cross-*layer* (not cross-*model*) routing contributions on four models and identifies "MoE entanglement" as a within-model phenomenon.

5. **Platonic Representation Hypothesis** (Huh et al., ICML 2024, arXiv:2405.07987) — the methodological template. They use **mutual k-NN alignment** and CKA on paired datasets to show vision and language kernels increasingly correlate with scale. Crucially, mutual k-NN sidesteps absolute coordinate systems: it asks "do model A and model B agree on *which inputs are similar*?" This nearest-neighbor framing is highly transferable to the present project.

**Verdict**: the present project's FGW-based cross-family routing comparison appears genuinely novel. The Myth paper and OLMoE provide partial benchmarks; A Closer Look provides the closest method analog. Nobody has measured per-expert correspondence across MoE *families*.

---

## 4. Mechanistic findings about routing gates

The router is mechanically a single linear projection P ∈ ℝ^{E×d} of the residual stream, followed by top-k softmax. Therefore by construction it can only "see" the linear projection of the hidden state into an E-dimensional subspace.

- **OpenMoE**: gates effectively memorize a token-ID → expert mapping early in training. Once fixed, the router's actual input dependence is weak.
- **The Myth paper**: gives the explicit upper bound ‖P·hᵢ − P·hⱼ‖₂ ≤ ‖P·Π_r‖₂·‖hᵢ−hⱼ‖ — routing distance is upper-bounded by hidden-state distance in the principal subspace of P.
- **RouterInterp / Superposed Specialisation Hypothesis** (Lasy et al., ICLR 2026): an expert is best modeled as a **disjoint union of SAE features** — not a single coherent concept. The most-predictive SAE features for an expert can be aggregated into a natural-language description that achieves 77% higher accuracy than the prior state of the art at predicting routing.
- **Wahib group, ICLR 2026**: previous MoE layer outputs contribute more to subsequent routing than attention outputs — routing is partly "internally driven" by other MoE layers, creating MoE-entanglement.
- **Iezzi et al. (POS sensitivity)**: gates demonstrably encode POS information — high POS-prediction accuracy from routing paths.

Together: gates attend to **a linear projection of features that bundle token identity + POS + (deeper) context**. The "token identity" signal is dominant in early layers; "context" emerges in deeper layers.

---

## 5. Implications for cross-model expert comparison

Given the project's constraint (per-expert metric, cross-model comparable, robust to "same type, different instance"), here are 5 candidate metrics ranked by appropriateness.

### Candidate A: Per-expert POS distribution histogram (fine-grained, not the 5-bin collapse)

- **Phenomenon**: syntactic specialization.
- **Literature support**: Iezzi et al. 2025 (POS routes predict POS strongly); Mixtral §5; ST-MoE.
- **Pros**: cross-model comparable (POS tagger is external); directly tested as a positive signal at finer granularity; addresses the "math-content weakness" because all math tokens become NUM/SYM/X regardless of instance.
- **Cons**: requires POS tagger; the user *already tried a coarse version and it failed* — needs the full UD POS tagset (~17 tags), not 5 bins.
- **Complexity**: low (rerun the existing pipeline with spaCy's full `pos_` not the collapsed bins).

### Candidate B: Per-expert token-identity distribution + tokenizer-agnostic alignment via embedding clustering

- **Phenomenon**: token-identity specialization (the strongest signal in OpenMoE, Expert Strikes Back).
- **Literature support**: OpenMoE (Context-Independent Specialization); Expert Strikes Back; OLMoE Vocabulary Specialization.
- **Method**: embed each token's string in a *shared* sentence/word embedding space (e.g., a small frozen encoder like SBERT or fasttext), then represent each expert by the *centroid + covariance* (or just centroid) of its top-routed tokens' embeddings.
- **Pros**: tokenizer-agnostic, captures "type" via embedding similarity not exact-string match, so "math content" from different prompts maps close.
- **Cons**: depends on choice of external embedder; conflates near-synonyms.
- **Complexity**: medium.

### Candidate C: Per-expert "next-token effect" — output unembedding direction

- **Phenomenon**: what an expert *promotes* in the vocabulary distribution.
- **Literature support**: OLMoE's unembedding-clustered domains; "Expert Strikes Back" output analysis.
- **Method**: compute mean output direction of expert e, project through that model's unembedding, get a vocabulary distribution → align across models via tokenizer-shared vocabulary or via embedding-space comparison.
- **Pros**: independent of which prompts you fed in (so the "math-content weakness" disappears: the expert is characterized by what it *outputs*, not what it *received*); directly grounded in the model's own functional contribution.
- **Cons**: requires running and capturing internal activations; tokenizer alignment for vocabulary distribution.
- **Complexity**: medium-high.

### Candidate D: Per-expert SAE-feature fingerprint (RouterInterp-style)

- **Phenomenon**: fine-grained feature-level specialization (Superposed Specialisation Hypothesis).
- **Literature support**: RouterInterp (Lasy et al. ICLR 2026); A Closer Look (neurons as fine-grained experts).
- **Pros**: highest interpretive fidelity; matches the strongest current theoretical framing.
- **Cons**: requires training an SAE per model (expensive); SAE features are model-specific — cross-model comparison requires a second alignment step (e.g., embedding feature descriptions in shared text-embedding space).
- **Complexity**: high.

### Candidate E: Per-expert pooled-activation embedding (Platonic-style, mutual k-NN)

- **Phenomenon**: representational geometry of the inputs the expert receives.
- **Literature support**: Platonic Representation Hypothesis (mutual k-NN); Myth paper (sequence-level cosine).
- **Method**: for each expert e, average the hidden states *into* the expert's projection over its top-routed tokens → get one vector per expert in the model's hidden-dim. Compare across models via mutual k-NN (does expert e's input set look like any expert in model B's?).
- **Pros**: directly testable against PRH; sidesteps tokenizer differences by working in continuous space; mutual k-NN sidesteps coordinate misalignment.
- **Cons**: aligns with the Myth paper's critique (you may just be measuring hidden-state geometry); requires a shared embedding space for "input tokens" or a careful k-NN over input identities.
- **Complexity**: medium.

---

## 6. Recommended metric

**Recommendation: Candidate C — Per-expert output unembedding fingerprint, compared cross-model via vocabulary embedding alignment.**

### Formal sketch

For each model M and each expert e at layer ℓ:
1. Compute mean *output* contribution v_e^M = mean over routed tokens of the expert's output residual.
2. Project through M's unembedding U^M: p_e^M = softmax(U^M · v_e^M) ∈ Δ^{|V^M|} — a vocabulary distribution over what this expert *promotes*.
3. Map vocabulary distributions to a shared semantic space: for each token t in V^M, embed its string with a *fixed external embedder* φ (e.g., a frozen SBERT). The expert's semantic fingerprint is the *expected token embedding* under p_e^M: f_e^M = Σ_t p_e^M(t) · φ(t) ∈ ℝ^{d_φ}.
4. Cross-model distance: cosine(f_e^M, f_{e'}^{M'}); or, following Platonic, mutual k-NN agreement on which experts each model considers semantically nearest.

### Why this addresses the "math-content weakness"

The fingerprint depends on what the expert *outputs* (promotes in the vocabulary distribution), not what *char-spans* fed into it. Two experts that both push probability mass toward mathematical operators / digits / "function" will have similar f, even if one saw `x^2+1` and the other saw `\\sum_i a_i`. The instance-level prompt-position information is washed out by integration over the expert's full token stream. This is exactly the type-vs-instance separation the user identified as missing.

### Why this addresses the Myth paper's critique

The Myth paper argues routing inherits hidden-state geometry; therefore *input-side* specialization metrics are confounded by representation geometry. The unembedding-output fingerprint is an *output-side* metric: it measures the expert's contribution to the model's functional output, not the geometry of its input. This is the only level at which "what does this expert *do*" is well-defined independent of the embedding manifold.

### Literature anchors

- **OLMoE §5.3** uses k-means on the unembedding to define domains natively — direct precedent for unembedding-based analysis.
- **A Closer Look into MoE** (Lo et al. 2024) finds that expert *output* similarity is the most cross-model-stable signal.
- **The Expert Strikes Back** anchors "expert-level" output interpretation.
- **Platonic Representation Hypothesis** provides the cross-model mutual k-NN comparison framework.
- **RouterInterp** validates that a single fingerprint per expert with a natural-language interpretation is achievable.

### What the paper would look like

The complementary-metric paper:
1. *Load* (existing) measures *whether* same-family models route similar volumes through corresponding experts — the **aggregate** signal.
2. *Unembedding fingerprint* (proposed) measures *whether* same-family models' corresponding experts **functionally promote** similar token semantics — the **specialization** signal.
3. Within FGW, replace the vertex-feature C with the cosine similarity matrix over f_e^M values, normalized to [0,1].
4. The predicted finding (consistent with PRH): same-family pairs have substantially higher FGW alignment when both feature matrices are used than when load alone is used; cross-family pairs are penalized further by output-fingerprint disagreement; the Mixtral×Qwen quadrant should now look even more anomalous.
5. The Myth paper's critique is addressed in a defense paragraph: by working in output (unembedding) space rather than input (residual) space, we sidestep the "geometry not domain" confound.

### Risk to anticipate at NeurIPS rebuttal

The fingerprint may *also* end up reflecting tokenizer/unembedding geometry rather than expert function. Defend by: (a) showing fingerprints from *random* experts (random output direction) are clearly separable from learned experts; (b) using mutual k-NN agreement on shared concept tokens (digits, common English words, punctuation) as a tokenizer-invariant sanity check; (c) running an ablation where φ is varied across embedders to confirm signal is embedder-agnostic.

---

## Sources

- [Mixtral of Experts (Jiang et al. 2024)](https://arxiv.org/pdf/2401.04088)
- [DeepSeekMoE (Dai et al. ACL 2024)](https://arxiv.org/html/2401.06066v1)
- [OLMoE (Muennighoff et al. 2024)](https://arxiv.org/html/2409.02060v1)
- [ST-MoE (Zoph et al. 2022)](https://arxiv.org/pdf/2202.08906)
- [A Closer Look into MoE in LLMs (Lo et al. 2024)](https://arxiv.org/html/2406.18219v2)
- [The Myth of Expert Specialization in MoEs](https://arxiv.org/html/2604.09780v1)
- [Do Domain-specific Experts exist in MoE-based LLMs?](https://arxiv.org/pdf/2604.05267)
- [Part-Of-Speech Sensitivity of Routers in MoE (Iezzi et al. COLING 2025)](https://arxiv.org/abs/2412.16971)
- [OpenMoE (Xue et al. ICML 2024)](https://arxiv.org/html/2402.01739v2)
- [Unveiling Super Experts in MoE LLMs (Su et al. 2025)](https://arxiv.org/html/2507.23279)
- [Multilingual Routing in Mixture-of-Experts](https://arxiv.org/html/2510.04694v1)
- [The Expert Strikes Back](https://arxiv.org/pdf/2604.02178)
- [RouterInterp: Superposed Specialisation in MoE Routing (Lasy et al. ICLR 2026)](https://openreview.net/forum?id=9a5i2vyMwN)
- [Understanding Cross-layer Contributions to MoE Routing in LLMs (Wahib group, ICLR 2026)](https://openreview.net/forum?id=BqyPLOkxFY)
- [Platonic Representation Hypothesis (Huh et al. 2024)](https://arxiv.org/abs/2405.07987)
- [MoE Routing Testbed](https://arxiv.org/pdf/2604.07030)
- [Survey on Mixture of Experts (Cai et al. 2024)](https://arxiv.org/html/2407.06204v3)
