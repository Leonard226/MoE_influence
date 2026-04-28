# Per-neuron score derivation

Extending §4.1 of the paper one level deeper, by substituting Eq. 11 (with the corrected upper bound $d_{ffn}$) into Eq. 10.

## Setup

Eq. 10 — per-expert score from a sending MoE layer $c$ to expert $n$ in receiving layer $\ell$:

$$S(g^{\ell,n}, m^c_{out,i}) = g^{\ell,n} \cdot \sum_{j \in J} \overline{LN}^\ell_i\!\big(r^{c,j}(m^c_{in,i})\, e^{c,j}_{out,i}\big)$$

Eq. 11 (corrected, relabeled to layer $c$, expert $j$) — neuron-level decomposition of the SwiGLU expert output:

$$e^{c,j}_{out,i} = \sum_{z=1}^{d_{ffn}} W_{d\,(:,z)}^{c,j}\, \alpha^{c,j}_z(m^c_{in,i})$$

with the scalar neuron activation

$$\alpha^{c,j}_z(m^c_{in,i}) := \sigma\!\big(W_{g\,(z,:)}^{c,j}\!\cdot m^c_{in,i}\big)\cdot\big(W_{u\,(z,:)}^{c,j}\!\cdot m^c_{in,i}\big)$$

So $\alpha^{c,j}_z$ is a scalar (the SwiGLU activation of neuron $z$) and $W_{d\,(:,z)}^{c,j} \in \mathbb{R}^{d_e}$ is the column it projects back through.

## Two facts we use

**(i) The approximate LN is linear in its argument.** From the paper,

$$\overline{LN}^\ell_i(v) = \frac{v\odot \gamma^\ell}{\rho^\ell_i}, \qquad \rho^\ell_i := \mathrm{RMS}\!\big(x^\ell_{in,i} + a^\ell_{out,i}\big)$$

The RMS denominator $\rho^\ell_i$ is a **scalar** (depends on $\ell, i$ but not on $v$), and $\gamma^\ell$ is a fixed parameter vector. So $\overline{LN}^\ell_i$ is linear: $\overline{LN}^\ell_i\!\big(\sum_k v_k\big) = \sum_k \overline{LN}^\ell_i(v_k)$, and scalar coefficients pass through.

**(ii) The router score is linear** in its right argument: $g^{\ell,n}\cdot\big(\sum_k v_k\big) = \sum_k g^{\ell,n}\cdot v_k$.

## Substitution

Plug Eq. 11 into Eq. 10, pull the scalar $r^{c,j}$ out of $\overline{LN}$, and apply linearity of both $\overline{LN}$ and the dot product:

$$S(g^{\ell,n}, m^c_{out,i}) = \sum_{j\in J} r^{c,j}\, g^{\ell,n}\cdot \overline{LN}^\ell_i\!\Big(\sum_{z=1}^{d_{ffn}} \alpha^{c,j}_z\, W_{d\,(:,z)}^{c,j}\Big)$$

$$= \sum_{j\in J}\sum_{z=1}^{d_{ffn}} r^{c,j}\, \alpha^{c,j}_z\,\Big[\,g^{\ell,n}\cdot \overline{LN}^\ell_i\!\big(W_{d\,(:,z)}^{c,j}\big)\,\Big]$$

The summand is the **score contributed by a single neuron** $(c, j, z)$ to expert $n$ in receiving layer $\ell$:

$$\boxed{\;S^{\ell,n}_{(c,j,z)} \;=\; r^{c,j}(m^c_{in,i})\,\cdot\, \alpha^{c,j}_z(m^c_{in,i})\,\cdot\, \big[\,g^{\ell,n}\cdot \overline{LN}^\ell_i\!\big(W_{d\,(:,z)}^{c,j}\big)\,\big]\;}$$

## Cleaner three-factor form

Expanding the LN approximation explicitly factors out the scalar $1/\rho^\ell_i$ and exposes a fully **static** (data-independent) alignment

$$A^{\ell,n}_{c,j,z} := g^{\ell,n}\cdot\big(W_{d\,(:,z)}^{c,j}\odot \gamma^\ell\big) \in \mathbb{R}$$

depending only on trained weights ($g^{\ell,n}$, $\gamma^\ell$, $W_d^{c,j}$). Then

$$S^{\ell,n}_{(c,j,z)}(m^c_{in,i}) \;=\; \frac{1}{\rho^\ell_i}\,\cdot\, \underbrace{r^{c,j}(m^c_{in,i})}_{\text{expert gate}}\,\cdot\, \underbrace{\alpha^{c,j}_z(m^c_{in,i})}_{\text{neuron activation}}\,\cdot\, \underbrace{A^{\ell,n}_{c,j,z}}_{\text{static alignment}}$$

Three multiplicative factors with very different character:

- $A^{\ell,n}_{c,j,z}$ — **static and precomputable** for every (receiving router, sending neuron) pair. Encodes how the down-projection direction of neuron $z$ aligns with router $(\ell, n)$ after $\gamma$-rescaling.
- $r^{c,j}$ — **sparse-dynamic**, nonzero only for the top-K experts at layer $c$.
- $\alpha^{c,j}_z$ — **dense-dynamic** but heavily attenuated by the SiLU gate term, so most neurons contribute little per token.
- $1/\rho^\ell_i$ — global per-token scalar shared across all $(c, j, z)$; irrelevant for *rankings* of contributions within a fixed receiving $(\ell, i)$.

## Sanity check

Summing over $z$ inside the bracket should reconstruct Eq. 10's per-expert term:

$$\sum_{z=1}^{d_{ffn}} \alpha^{c,j}_z\, A^{\ell,n}_{c,j,z} = g^{\ell,n}\cdot\Big[\Big(\sum_z \alpha^{c,j}_z W_{d\,(:,z)}^{c,j}\Big)\odot \gamma^\ell\Big] = g^{\ell,n}\cdot\big(e^{c,j}_{out,i}\odot \gamma^\ell\big) = \rho^\ell_i \cdot g^{\ell,n}\cdot \overline{LN}^\ell_i(e^{c,j}_{out,i}) \quad\checkmark$$

So the decomposition is internally consistent with Eq. 10.

## Why the authors probably punted

For OLMoE alone, the static tensor $A^{\ell,n}_{c,j,z}$ has shape

$$\text{(receiving } (\ell, n)\text{)} \times \text{(sending } (c, j, z)\text{)} \approx (16 \cdot 64) \times (16 \cdot 64 \cdot 1024) \approx 10^9$$

entries. For Mixtral-8x7B it is much larger ($d_{ffn} = 14336$). Per-token neuron-level variance maps are tractable, but the full cross-layer neuron-to-router contribution map is huge. The factorization above is what makes attacking it feasible — the static piece $A$ is computed once and reused, and only the cheap scalar factors $r^{c,j}\,\alpha^{c,j}_z$ vary per token.

## Possible next experiments (extending §5–§7 to the neuron level)

1. **Neuron-level scoring variance maps** (analog of Fig. 3) to find "super-neurons" inside super-experts.
2. **Static-vs-dynamic ablation:** how much of an expert's downstream variance is explained by $A$ alone vs. by the dynamic $\alpha$ activations.
3. **Neuron-level AARV** (analog of Fig. 6) on M1E9 in OLMoE, to localize *which* of its 1024 neurons drive the M1→M5/M10/M15 stripes.
4. **Sparsity / pruning:** neurons with consistently small $|A|$ contribute negligibly to routing regardless of activation — candidates for compression that preserve routing behavior.
