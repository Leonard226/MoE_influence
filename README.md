# Per-neuron routing influence in OLMoE

Code for the per-neuron extension of the per-expert routing decomposition,
applied to OLMoE-1B-7B-0924. The writeup lives in `69f00410eda46469665d3f0c/main.tex`
and reads three figures produced here: `tier1_super_neurons.pdf`,
`tier2_aarv.pdf`, `tier2_per_neuron_aarv.pdf`.

## Layout

```
config.yaml                    result_path for all artifacts
customized_models/             OlmoeForCausalLM with per-neuron hooks
dataset/                       C4 prompt loader
experiments/
  variance/                    correlative metric (Figure 1)
    static_alignment.py          one-time precompute of V[c,j,z,l] = Var_n A^{l,n}_{c,j,z}
    per_expert.py                per-expert variance score from forward passes
    per_neuron.py                per-neuron variance score (uses V from static_alignment)
    plot.py                      Figure 1 (3 panels)
  ablation/                    causal metric (Figures 2 & 3)
    named_ablations.py           AARV for whole-expert and dominant-neuron ablations
    per_neuron_sweep.py          AARV for every neuron in each named expert (1024 / expert)
    plot_named.py                Figure 2
    plot_per_neuron.py           Figure 3
```

`models_backup/` and `modified_models_backup/` are archived snapshots and must not be modified.
`tools/` is auxiliary and not required to reproduce the experiments.

## Reproduction

Set `result_path` in `config.yaml`. Then, from the project root:

```bash
# Variance metric → Figure 1
python experiments/variance/static_alignment.py     # writes V.pt
python experiments/variance/per_expert.py           # writes score_variance_per_expert.pt
python experiments/variance/per_neuron.py           # writes T_dyn.pt
python experiments/variance/plot.py                 # writes tier1_super_neurons.pdf

# Ablation metric → Figures 2 and 3
python experiments/ablation/named_ablations.py      # writes aarv.pt, aarv_summary.json
python experiments/ablation/per_neuron_sweep.py     # writes per_neuron_aarv.pt
python experiments/ablation/plot_named.py           # writes tier2_aarv.pdf
python experiments/ablation/plot_per_neuron.py      # writes tier2_per_neuron_aarv.pdf
```

All artifacts (`.pt`, `.json`, `.pdf`, `.png`) land in
`{result_path}/variance/` and `{result_path}/ablation/`.
