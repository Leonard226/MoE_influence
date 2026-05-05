"""Plot per-neuron token specialization as a 2x3 grid of stem plots. Each
panel shows the top-100 tokens by mean activation for one named neuron;
the top trigger tokens are labelled and the title quantifies how much of
the neuron's total activation mass lives in its top-2 tokens. Also writes
a JSON of the top-50 trigger tokens for the appendix.
"""
import json
import os
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from transformers import AutoTokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    config = yaml.safe_load(f)
art = os.path.join(config["result_path"], "specialization")

mpl.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "serif",
    "font.size": 10,
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 9,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.7,
    "xtick.major.width": 0.7,
    "ytick.major.width": 0.7,
})

C_M1E9   = "#d62728"
C_M2E30  = "#2ca02c"
C_M4E14  = "#1f77b4"
C_M14E60 = "#9467bd"
EXPERT_COLOR = {1: C_M1E9, 2: C_M2E30, 4: C_M4E14, 14: C_M14E60}

MIN_COUNT = 20
N_SHOW = 100      # how many ranks to display in the stem plot
TOP_K_JSON = 50   # how many to dump to JSON

# How many top tokens to label per panel.
N_LABEL = {
    "M1E9_z915":   3,
    "M2E30_z742":  3,
    "M4E14_z391":  3,
    "M4E14_z336":  3,
    "M4E14_z956":  3,
    "M14E60_z357": 3,
}
# Panels where the labels should sit above the stem tops (for category specialists
# whose top stems are all close in height, so labels at stem-top would collide).
LABELS_ABOVE = {"M2E30_z742", "M14E60_z357"}

data = torch.load(os.path.join(art, "per_token_activation.pt"))
tokenizer = AutoTokenizer.from_pretrained("allenai/OLMoE-1B-7B-0924")

NAMES = list(data.keys())
fig, axs = plt.subplots(2, 3, figsize=(13, 6.6),
                         gridspec_kw={"wspace": 0.30, "hspace": 0.45})

top_lists_json = {}

for ax, name in zip(axs.flat, NAMES):
    info = data[name]
    s    = info["sum"].numpy()
    cnt  = info["count"].numpy()
    color = EXPERT_COLOR.get(info["c"], "gray")

    mean = np.zeros_like(s)
    valid = cnt >= MIN_COUNT
    mean[valid] = s[valid] / cnt[valid]

    sorted_idx = np.argsort(mean)[::-1]
    n_active = int((mean > 0).sum())
    n_show = min(N_SHOW, n_active)

    ranks = np.arange(1, n_show + 1)
    values = mean[sorted_idx[:n_show]]

    # Fraction of total activation captured by the top-3 tokens.
    total_mass = s.sum()
    top3_frac = float(s[sorted_idx[:3]].sum() / total_mass) if total_mass > 0 else 0.0

    # Save top-50 JSON.
    top_lists_json[name] = [
        {
            "tok_id": int(t),
            "decoded": tokenizer.decode([int(t)]),
            "mean_act": float(mean[t]),
            "count": int(cnt[t]),
        }
        for t in sorted_idx[:TOP_K_JSON]
    ]

    # Stem plot: a vertical line per token + a marker at the top.
    ax.vlines(ranks, 0, values, colors=color, lw=1.2, alpha=0.85)
    ax.scatter(ranks, values, color=color, s=10, zorder=5, edgecolor="none")

    # Highlight + label the top-N tokens for this panel.
    n_highlight = N_LABEL.get(name, 2)
    ax.scatter(ranks[:n_highlight], values[:n_highlight],
               color=color, s=55, zorder=6, edgecolor="white", linewidth=1.0)

    y_max = max(values[0] * 1.25, 0.1)
    above = name in LABELS_ABOVE

    for i in range(n_highlight):
        token = tokenizer.decode([int(sorted_idx[i])]).replace("\n", "\\n")
        if above:
            # Stack labels in the upper region so they don't collide with each other or with stems.
            label_x = ranks[i] + 12
            label_y = y_max * (0.93 - 0.10 * i)
        else:
            label_x = ranks[i] + 8
            # Per-(panel, rank) lift overrides for visual cleanup.
            lift_override = (name, i) in {
                ("M4E14_z391", 0),
                ("M4E14_z391", 2),
                ("M4E14_z956", 1),
            }
            if lift_override:
                label_y = values[i] + y_max * 0.10
            elif values[i] < y_max * 0.10:
                # Lift small values so they don't sit on top of the low-stem carpet.
                label_y = max(values[i] + y_max * 0.08, y_max * 0.12)
            else:
                label_y = values[i] * 0.97
        ax.annotate(repr(token),
                    xy=(ranks[i], values[i]),
                    xytext=(label_x, label_y),
                    fontsize=9.5, va="center", ha="left",
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.5))

    # In-panel annotation of the top-2 mass concentration.
    ax.text(0.97, 0.12,
            f"top-3 tokens account for\n{top3_frac*100:.1f}% of total activation",
            transform=ax.transAxes, ha="right", va="bottom",
            fontsize=9, color="0.25",
            bbox=dict(facecolor="white", edgecolor="0.8", linewidth=0.6, pad=3, alpha=0.9))

    ax.set_xlim(-1, N_SHOW + 5)
    ax.set_ylim(0, y_max)
    ax.set_xlabel(f"Token rank (top-{N_SHOW})")
    ax.set_ylabel("Mean activation")
    # All four named experts are high-variance experts; the first three concentrate the
    # variance in a small set of neurons, M14E60 distributes it across many.
    expert_name = name.split("_")[0]
    if expert_name == "M14E60":
        suffix = " (high-variance, distributed)"
    else:
        suffix = " (high-variance, concentrated)"
    ax.set_title(f"{name}{suffix}")
    ax.grid(True, alpha=0.3, lw=0.4, axis="y")
    ax.set_axisbelow(True)

fig.tight_layout()

out_pdf = os.path.join(art, "token_specialization.pdf")
out_png = os.path.join(art, "token_specialization.png")
fig.savefig(out_pdf, format="pdf", bbox_inches="tight")
fig.savefig(out_png, format="png", bbox_inches="tight", dpi=180)
plt.close(fig)
print(f"Wrote {out_pdf}")

with open(os.path.join(art, "top_tokens.json"), "w") as f:
    json.dump(top_lists_json, f, indent=2, ensure_ascii=False)
print(f"Wrote {os.path.join(art, 'top_tokens.json')}")

# Stdout summary.
print(f"\nTop-5 trigger tokens (min count = {MIN_COUNT}):")
for name in NAMES:
    print(f"\n{name}:")
    for entry in top_lists_json[name][:5]:
        d = entry["decoded"].replace("\n", "\\n")
        print(f"  {repr(d):>20}  mean={entry['mean_act']:.3f}  n={entry['count']}")
