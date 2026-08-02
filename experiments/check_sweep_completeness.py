"""Completeness check for the single-node ablation/super-weight/routing-
shift sweep -- no torch dependency, safe to run anywhere the results/
directory is visible (this Mac or the cluster).

For each model, checks three things:
  1. ablation/{model}_c4.json has every label from ablate_experts.py's
     DEFAULT_TARGETS (both singles and joint sets).
  2. routing_shift/{model}_c4.json has an entry for every non-baseline
     label actually present in the ablation JSON (so newly-added ablation
     targets are caught even if DEFAULT_TARGETS was edited after the last
     routing-shift run).
  3. super_weights/{model}_c4.json's global (layer, expert) grid coverage,
     broken down by refined / screened / legacy (pre-peak-filter) entries.
     <100% here is not automatically a problem -- experts with zero routed
     tokens in the calibration set are legitimately absent -- so this is
     reported for eyeballing, not treated as a hard failure.

DEFAULT_TARGETS is pulled out of ablate_experts.py via source-text
extraction (not import), specifically so this script never needs torch.

Usage:
    python experiments/check_sweep_completeness.py
    python experiments/check_sweep_completeness.py --models mixtral-8x22b,deepseek-v2-lite
"""
import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Mirrors find_super_weights.py's MODELS registry (n_experts, moe_layers)
# without importing it (that file itself requires torch to import, since
# it's not guarded behind __main__ for its own imports).
SW_MODELS = {
    "olmoe": {"n_experts": 64, "moe_layers": list(range(16))},
    "mixtral-8x7b": {"n_experts": 8, "moe_layers": list(range(32))},
    "mixtral-8x22b": {"n_experts": 8, "moe_layers": list(range(56))},
    "phi-3.5-moe": {"n_experts": 16, "moe_layers": list(range(32))},
    "qwen3-30b-a3b": {"n_experts": 128, "moe_layers": list(range(48))},
    "deepseek-v2-lite": {"n_experts": 64, "moe_layers": list(range(1, 27))},
}


def load_default_targets() -> dict[str, str]:
    src = (ROOT / "experiments/ablate_experts.py").read_text()
    m = re.search(r"DEFAULT_TARGETS = \{.*?\n\}", src, re.DOTALL)
    ns: dict = {}
    exec(m.group(0), ns)
    return ns["DEFAULT_TARGETS"]


def parse_targets(spec: str) -> list[list[tuple[int, int]]]:
    runs = []
    for run in spec.split(";"):
        group = []
        for lab in run.strip().split("+"):
            lab = lab.strip().upper()
            l_str, e_str = lab.lstrip("L").split("E")
            group.append((int(l_str), int(e_str)))
        runs.append(group)
    return runs


def label_of(group: list[tuple[int, int]]) -> str:
    return "+".join(f"L{l}E{e}" for l, e in group)


def check_model(model: str, default_targets: dict[str, str]) -> bool:
    """Returns True if a problem was found."""
    print(f"=== {model} ===")
    problem = False

    ab_path = RESULTS / "ablation" / f"{model}_c4.json"
    sw_path = RESULTS / "super_weights" / f"{model}_c4.json"
    rs_path = RESULTS / "routing_shift" / f"{model}_c4.json"

    if not ab_path.exists():
        print("  MISSING ablation file entirely")
        return True
    ab = json.loads(ab_path.read_text())

    # 1. ablation completeness vs DEFAULT_TARGETS
    expected = {"baseline"}
    for group in parse_targets(default_targets[model]):
        expected.add(label_of(group))
    missing_ab = sorted(expected - set(ab.keys()))
    if missing_ab:
        print(f"  [ablation] MISSING {len(missing_ab)} expected labels: {missing_ab}")
        problem = True
    else:
        print(f"  [ablation] OK -- all {len(expected)} DEFAULT_TARGETS labels present "
              f"({len(ab)} total incl. random controls)")

    # 2. routing_shift completeness vs actual ablation keys
    if not rs_path.exists():
        print("  [routing_shift] MISSING file entirely")
        problem = True
    else:
        rs = json.loads(rs_path.read_text())
        ab_keys = set(ab.keys()) - {"baseline"}
        missing_rs = sorted(ab_keys - set(rs.keys()))
        if missing_rs:
            print(f"  [routing_shift] MISSING {len(missing_rs)} of {len(ab_keys)} "
                  f"ablation labels: {missing_rs}")
            problem = True
        else:
            print(f"  [routing_shift] OK -- all {len(ab_keys)} ablation labels present")

    # 3. super_weights global grid coverage
    if not sw_path.exists():
        print("  [super_weights] MISSING file entirely")
        problem = True
    else:
        sw = json.loads(sw_path.read_text())
        cfg = SW_MODELS[model]
        full_grid = {(l, e) for l in cfg["moe_layers"] for e in range(cfg["n_experts"])}
        present = {(l, e) for (l, e) in full_grid if f"L{l}E{e}" in sw}
        n_refined = sum(1 for v in sw.values() if v.get("refined"))
        n_screened = sum(1 for v in sw.values() if v.get("refined") is False)
        n_legacy = sum(1 for v in sw.values() if "refined" not in v)
        coverage = len(present) / len(full_grid) * 100 if full_grid else 0.0
        print(f"  [super_weights] {len(present)}/{len(full_grid)} ({coverage:.1f}%) of the "
              f"full (layer,expert) grid present -- {n_refined} refined, {n_screened} "
              f"screened, {n_legacy} legacy (pre-filter, always refined)")

    return problem


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--models", default=None,
                   help="Comma-separated model keys (default: all 6).")
    args = p.parse_args()

    default_targets = load_default_targets()
    models = args.models.split(",") if args.models else list(SW_MODELS)

    any_problem = False
    for model in models:
        any_problem |= check_model(model.strip(), default_targets)

    print()
    print("PROBLEMS FOUND -- see above" if any_problem else "all checks passed")


if __name__ == "__main__":
    main()
