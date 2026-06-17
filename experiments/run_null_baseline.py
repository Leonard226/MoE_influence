"""Null-baseline FGW sweep on c4.

For each (alpha, Q) cell, compute four null distributions of FGW similarity:
  1. trained_vs_random_same    -- 5 archs x 3 seeds = 15 pairs
  2. trained_vs_random_cross   -- 5 x 4 archs x 3 seeds = 60 pairs
  3. random_vs_random_cross    -- C(5,2) x 3 x 3 = 90 pairs
  4. random_vs_random_same     -- 5 archs x C(3,2) = 15 pairs   (seed variance)

Total per (alpha, Q): 180 pairs x 3 alpha = 540 FGW calls.
Total over (alpha, Q): 1620 FGW calls. ~hours on a single CPU node.

SLURM-array friendly: --q-idx selects one of the 3 quantile values; each task
builds its 20 triples (5 archs x [trained + 3 seeds]) once for its Q and runs
all 540 FGW calls for that Q sequentially. Submit as a 3-task array; the
results across Q are concatenated by --merge.

Settings match the headline sweep:
  alpha in {0, 0.5, 1},  Q in {0.9, 0.99, 0.999},  beta = 0.5,
  n_init = 5,  act_norm = log_max,  load_norm = log_max.

Output:
  ${result_path}/circuits/null_baseline/null_S_Q{q}.csv     (per-Q task)
  ${result_path}/circuits/null_baseline/null_S.csv          (after --merge)

Each CSV row: comparison, arch_i, seed_i, arch_j, seed_j, alpha, Q, S, n_verts_i, n_verts_j.
seed_i / seed_j are None for the trained side of a pair.
"""
from __future__ import annotations

import argparse
import csv
import itertools
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch
import yaml

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")
torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.fgw import build_triple, fgw_similarity                # noqa: E402
from experiments.run_alpha_beta_sweep import (                          # noqa: E402
    _edge_quantile_threshold, _subset_triple,
)


# --- Configuration --------------------------------------------------------
ARCHS = [
    "mixtral-8x7b",
    "olmoe",
    "deepseek-v2-lite",
    "qwen3-30b-a3b",
    "phi-3.5-moe",
]
SEEDS = [0, 1, 2]
ALPHAS = [0.0, 0.5, 1.0]
QUANTILES = [0.9, 0.99, 0.999]
BETA = 0.5
N_INIT = 5
TASK = "c4"
ACT_NORM = "log_max"
LOAD_NORM = "log_max"
EDGE_TENSOR = "W_softmax"

OUT_SUBDIR = "null_baseline"


# --- I/O helpers ----------------------------------------------------------
def _dag_path(circuits_dir: Path, arch: str, seed: int | None) -> Path:
    if seed is None:
        return circuits_dir / f"dag_{arch}_{TASK}.pt"
    return circuits_dir / f"dag_{arch}_{TASK}_rand_s{seed}.pt"


def _classification_path(circuits_dir: Path, arch: str) -> Path:
    return circuits_dir / "classifications" / f"classify_{arch}_{TASK}.pkl"


def _build_triple_at_Q(dag_path: Path, classification: dict, Q: float):
    """Build the (C, F, mass, meta) triple at the given Q, with the same
    isolated-vertex filter the headline sweep uses."""
    dag = torch.load(dag_path, weights_only=False, map_location="cpu")
    W = dag[EDGE_TENSOR].float()
    L = W.shape[0]

    threshold = _edge_quantile_threshold(W, Q)

    triple = build_triple(
        dag, classification,
        beta=BETA, edge_threshold=threshold, edge_tensor=EDGE_TENSOR,
        act_norm_method=ACT_NORM, load_norm_method=LOAD_NORM,
    )

    # Isolated-vertex filter: drop vertices with no surviving forward edge.
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    survive = (torch.abs(W) > threshold) & fwd
    out_sparse = survive.sum(dim=(2, 3)).reshape(-1).cpu().numpy()
    in_sparse  = survive.sum(dim=(0, 1)).reshape(-1).cpu().numpy()
    keep_mask = (out_sparse > 0) | (in_sparse > 0)

    triple = _subset_triple(triple, keep_mask)
    triple[3]["quantile"] = Q
    triple[3]["edge_threshold"] = threshold

    del dag, W, survive, fwd
    return triple


# --- Pair enumeration ----------------------------------------------------
def _enumerate_pairs(comparison: str):
    """Yield ((arch_i, seed_i), (arch_j, seed_j)) pairs for one comparison
    type. seed_i / seed_j is None for the trained side."""
    if comparison == "trained_vs_random_same":
        for arch in ARCHS:
            for s in SEEDS:
                yield (arch, None), (arch, s)
    elif comparison == "trained_vs_random_cross":
        for arch_i in ARCHS:
            for arch_j in ARCHS:
                if arch_i == arch_j:
                    continue
                for s in SEEDS:
                    yield (arch_i, None), (arch_j, s)
    elif comparison == "random_vs_random_cross":
        for ai, arch_i in enumerate(ARCHS):
            for arch_j in ARCHS[ai + 1:]:
                for si in SEEDS:
                    for sj in SEEDS:
                        yield (arch_i, si), (arch_j, sj)
    elif comparison == "random_vs_random_same":
        for arch in ARCHS:
            for si, sj in itertools.combinations(SEEDS, 2):
                yield (arch, si), (arch, sj)
    else:
        raise ValueError(f"unknown comparison: {comparison}")


COMPARISONS = [
    "trained_vs_random_same",
    "trained_vs_random_cross",
    "random_vs_random_cross",
    "random_vs_random_same",
]


# --- Main per-Q routine ---------------------------------------------------
def _run_one_Q(Q: float, circuits_dir: Path, out_path: Path,
               n_init: int = N_INIT) -> None:
    """Build triples for one Q and run all 540 FGW calls."""
    # Build all 20 triples for this Q in sequence (memory-bounded).
    print(f"\n=== Q = {Q} ===", flush=True)
    triples: dict[tuple[str, int | None], tuple] = {}
    t_build = time.time()
    for arch in ARCHS:
        cls_path = _classification_path(circuits_dir, arch)
        if not cls_path.exists():
            raise FileNotFoundError(f"missing classification: {cls_path}")
        with open(cls_path, "rb") as f:
            cls = pickle.load(f)
        # Trained triple.
        tri = _build_triple_at_Q(_dag_path(circuits_dir, arch, None), cls, Q)
        triples[(arch, None)] = tri
        print(f"  built ({arch}, trained) @Q={Q}  n_verts={tri[3]['n_verts']:6d}  "
              f"({time.time() - t_build:6.1f}s)", flush=True)
        # Random triples.
        for s in SEEDS:
            tri = _build_triple_at_Q(_dag_path(circuits_dir, arch, s), cls, Q)
            triples[(arch, s)] = tri
            print(f"  built ({arch}, s={s})    @Q={Q}  n_verts={tri[3]['n_verts']:6d}  "
                  f"({time.time() - t_build:6.1f}s)", flush=True)

    # Run FGW calls.
    rows: list[dict] = []
    n_done, n_failed = 0, 0
    t0 = time.time()
    for comparison in COMPARISONS:
        pairs = list(_enumerate_pairs(comparison))
        print(f"\n--- {comparison}: {len(pairs)} pairs, "
              f"{len(ALPHAS)} alphas = {len(pairs) * len(ALPHAS)} FGW calls ---",
              flush=True)
        for pi, ((a_i, s_i), (a_j, s_j)) in enumerate(pairs):
            t_i = triples[(a_i, s_i)]
            t_j = triples[(a_j, s_j)]
            for alpha in ALPHAS:
                try:
                    S, _ = fgw_similarity(t_i, t_j, alpha=alpha, n_init=n_init)
                except Exception as e:
                    print(f"  FAIL  {comparison}  {a_i}/{s_i} x {a_j}/{s_j}  "
                          f"alpha={alpha}  Q={Q}  : {type(e).__name__}: "
                          f"{str(e)[:160]}", flush=True)
                    S = float("nan")
                    n_failed += 1
                rows.append({
                    "comparison": comparison,
                    "arch_i": a_i, "seed_i": "" if s_i is None else int(s_i),
                    "arch_j": a_j, "seed_j": "" if s_j is None else int(s_j),
                    "alpha": float(alpha), "Q": float(Q), "S": float(S),
                    "n_verts_i": int(t_i[3]["n_verts"]),
                    "n_verts_j": int(t_j[3]["n_verts"]),
                })
                n_done += 1
            # Checkpoint after each pair finishes its alpha sweep.
            if (pi + 1) % 5 == 0 or pi == len(pairs) - 1:
                _save_rows(out_path, rows)
                print(f"  [{comparison}]  pair {pi + 1:3d}/{len(pairs)}  "
                      f"n_done={n_done}  failed={n_failed}  "
                      f"({time.time() - t0:.0f}s elapsed)", flush=True)

    _save_rows(out_path, rows)
    print(f"\n=== Q={Q} done in {time.time() - t0:.0f}s "
          f"({n_done} cells, {n_failed} failures) ===", flush=True)


def _save_rows(out_path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["comparison", "arch_i", "seed_i", "arch_j", "seed_j",
                  "alpha", "Q", "S", "n_verts_i", "n_verts_j"]
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for row in rows:
            w.writerow(row)


# --- Merge step ----------------------------------------------------------
def _merge(out_dir: Path) -> None:
    parts = sorted(out_dir.glob("null_S_Q*.csv"))
    if not parts:
        print(f"ERROR: no per-Q CSV files in {out_dir}")
        sys.exit(1)
    merged = out_dir / "null_S.csv"
    fieldnames = None
    n_rows = 0
    with open(merged, "w", newline="") as fout:
        writer = None
        for p in parts:
            with open(p) as fin:
                reader = csv.DictReader(fin)
                if writer is None:
                    fieldnames = reader.fieldnames
                    writer = csv.DictWriter(fout, fieldnames=fieldnames)
                    writer.writeheader()
                for row in reader:
                    writer.writerow(row)
                    n_rows += 1
            print(f"  merged {p.name}", flush=True)
    print(f"\nMerged {len(parts)} files -> {merged} ({n_rows} rows)")


# --- Entry point ---------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--q-idx", type=int, default=None,
                        help="Index into QUANTILES (0=0.9, 1=0.99, 2=0.999). "
                             "Defaults to $SLURM_ARRAY_TASK_ID. Required unless "
                             "--merge is set.")
    parser.add_argument("--n-init", type=int, default=N_INIT)
    parser.add_argument("--merge", action="store_true",
                        help="Concatenate the three per-Q CSVs into null_S.csv.")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    circuits_dir = Path(result_path) / "circuits"
    out_dir = circuits_dir / OUT_SUBDIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.merge:
        _merge(out_dir)
        return

    q_idx = args.q_idx
    if q_idx is None:
        q_idx = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if not (0 <= q_idx < len(QUANTILES)):
        print(f"ERROR: --q-idx={q_idx} out of range [0, {len(QUANTILES)})")
        sys.exit(1)
    Q = QUANTILES[q_idx]
    out_path = out_dir / f"null_S_Q{Q:g}.csv"

    print(f"Null-baseline sweep")
    print(f"  Q              : {Q}  (--q-idx={q_idx})")
    print(f"  archs          : {ARCHS}")
    print(f"  seeds          : {SEEDS}")
    print(f"  alphas         : {ALPHAS}")
    print(f"  beta, n_init   : {BETA}, {args.n_init}")
    print(f"  act_norm       : {ACT_NORM}")
    print(f"  load_norm      : {LOAD_NORM}")
    print(f"  out_path       : {out_path}")

    _run_one_Q(Q, circuits_dir, out_path, n_init=args.n_init)


if __name__ == "__main__":
    main()
