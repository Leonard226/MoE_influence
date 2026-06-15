"""Feature ablation sweep at alpha = 0.

Per-vertex feature matrix F has 6 features (10 columns):
   col 0    : depth
   col 1    : tilde_out
   col 2    : tilde_in
   col 3    : load
   col 4    : tilde_act
   cols 5-9 : class_hist  (T = 5 macro classes; cf. main.tex)

We run a leave-one-out (LOO) ablation: for each feature, zero out the
corresponding column(s) of F (in both source and target) and compute
FGW at alpha = 0 across all same-task cross-model pairs at every Q.

Settings:
  alpha = 0 (pure feature -- ablation is meaningful here)
  beta  = 0.5 (matches the headline sweep)
  Q     in {0.9, 0.99, 0.999}
  pairs : same-task cross-model unordered pairs (28 per task * 8 tasks = 224)
  ablations: full + 6 LOO = 7 configurations
  n_init  : 5 (matches headline)

Total: 7 ablations x 3 Q x 224 pairs = 4704 FGW calls.

Helpers (q_threshold, _isolated_keep_mask, _build_filtered_triple) are
copied verbatim from fgw_alpha1_beta0_sweep.py to avoid the torch.quantile
trap on Qwen3-235B (~72M forward edges).

Output:
  <result_path>/circuits/feature_ablation/S_loo.npz
    S       : shape (7, 3, 8, 8, 8)  indexed (abl, q, task, m_i, m_j)
              S[a, q, t, i, j] = similarity for ablation a, quantile q,
              task t, model pair (i, j) with i != j. Diagonal NaN.
  <result_path>/circuits/feature_ablation/loo_summary.json
"""
from __future__ import annotations

import argparse
import copy
import json
import multiprocessing as mp
import os
import pickle
import sys
import time
from collections import OrderedDict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(var, "1")

import numpy as np
import torch
import yaml

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.fgw import build_triple, fgw_distance  # noqa: E402
from experiments.run_alpha_beta_sweep import _subset_triple  # noqa: E402


# ---------------------------------------------------------------------------
# Config.
# ---------------------------------------------------------------------------
MODELS = [
    "mixtral-8x7b",
    "mixtral-8x22b",
    "deepseek-v2-lite",
    "deepseek-v2",
    "qwen3-30b-a3b",
    "qwen3-235b-a22b",
    "olmoe",
    "phi-3.5-moe",
]
TASKS = [
    "c4", "math", "code",
    "wikitext2", "gsm8k", "humaneval",
    "pile-arxiv", "pile-github",
]
QUANTILES = [0.9, 0.99, 0.999]
ALPHA = 0.0
BETA = 1.0          # IRRELEVANT at alpha=0 (Wasserstein-only, C unused); but
                    # beta=1.0 skips the all-pairs shortest-path computation
                    # in build_triple (see fgw.py:342-346) -- pure performance
                    # optimisation, identical FGW output as beta=0.5 here.
N_INIT = 5          # matches headline sweep

# Activation feature normalisation. Set from --act-norm CLI flag in main().
# Defaults to "rank" so existing scripts and the per-pair-task workers behave
# identically to before. With "log_max" the script writes to a separate output
# directory (see main()) so the legacy results are preserved untouched.
_ACT_NORM_METHOD = "rank"

# Load feature normalisation. Set from --load-norm CLI flag in main().
# Defaults to "raw" (legacy: load = n_tok / mean_in_layer, range up to N).
# With "log_max" the per-layer log-max normalisation is used (range [0, 1]).
_LOAD_NORM_METHOD = "raw"

# Feature column indices in F (from experiments/fgw.py:324-330).
#   depth, out, in, load, act = single columns (0..4)
#   class_hist = 5 columns (5..9), the per-token-class histogram
FEATURE_COLS = {
    "depth": [0],
    "out":   [1],
    "in":    [2],
    "load":  [3],
    "act":   [4],
    "class": [5, 6, 7, 8, 9],
}

# Ablation list: full (no zeroing) + 6 LOO ablations.
# IMPORTANT: keep "full" at index 0 -- the analysis uses it as the baseline.
ABLATION_NAMES = [
    "full",
    "no_depth",
    "no_out",
    "no_in",
    "no_load",
    "no_act",
    "no_class",
]

# Family map for within/cross-family bookkeeping.
FAMILIES = {
    "Mixtral":  ["mixtral-8x7b", "mixtral-8x22b"],
    "DeepSeek": ["deepseek-v2-lite", "deepseek-v2"],
    "Qwen3":    ["qwen3-30b-a3b", "qwen3-235b-a22b"],
}
MODEL_TO_FAMILY = {m: f for f, ms in FAMILIES.items() for m in ms}

CACHE_DIR_NAME = "classifications"   # under {result_path}/circuits/


# ---------------------------------------------------------------------------
# Helpers (verbatim from fgw_alpha1_beta0_sweep.py except where noted).
# ---------------------------------------------------------------------------
def q_threshold(dag: dict, Q: float) -> float:
    """Per-graph Q-quantile of positive forward |W_softmax| edges.
    Uses np.quantile (torch errors above ~16M elements; Qwen3-235B has ~72M)."""
    if Q == 0.0:
        return 0.0
    W = dag["W_softmax"]
    L = W.shape[0]
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    edge_vals = torch.abs(W[fwd]).cpu().numpy().astype(np.float64)
    nz = edge_vals[edge_vals > 0]
    return float(np.quantile(nz, Q)) if len(nz) else 0.0


def _isolated_keep_mask(dag: dict, theta: float) -> np.ndarray:
    """Drop vertices that have zero surviving forward in- or out-edges above
    threshold. Same rule as run_alpha_beta_sweep._isolated_keep_mask."""
    W = dag["W_softmax"]
    L = W.shape[0]
    s_idx = torch.arange(L).view(-1, 1, 1, 1)
    r_idx = torch.arange(L).view(1, 1, -1, 1)
    fwd = (s_idx < r_idx).expand_as(W)
    survive = (torch.abs(W) > theta) & fwd
    out_sparse = survive.sum(dim=(2, 3)).reshape(-1).cpu().numpy()
    in_sparse  = survive.sum(dim=(0, 1)).reshape(-1).cpu().numpy()
    return (out_sparse > 0) | (in_sparse > 0)


def _build_filtered_triple(dag: dict, classification, theta: float,
                           act_norm_method: str = "rank",
                           load_norm_method: str = "raw"):
    """Build triple at BETA, edge_threshold=theta, then drop isolated vertices.
    PASSES the classification through so class_hist is meaningful (the alpha=0
    Wasserstein term needs it). Matches run_alpha_beta_sweep.build_triple_at_Q.
    `act_norm_method` / `load_norm_method` select the normalisation of the
    activation / load features; plumbed through to fgw.build_triple."""
    triple = build_triple(dag, classification, beta=BETA, edge_threshold=theta,
                          act_norm_method=act_norm_method,
                          load_norm_method=load_norm_method)
    keep_mask = _isolated_keep_mask(dag, theta)
    return _subset_triple(triple, keep_mask), int(keep_mask.sum())


# ---------------------------------------------------------------------------
# Feature ablation (the only genuinely new logic).
# ---------------------------------------------------------------------------
def _apply_ablation(triple, ablation_name: str):
    """Return a NEW triple with the F columns for `ablation_name` zeroed out.
    C and mass are unchanged. The original triple is not modified."""
    C, F, mass, meta = triple
    if ablation_name == "full":
        return triple   # no change needed
    feat_to_drop = ablation_name.replace("no_", "")
    cols = FEATURE_COLS[feat_to_drop]
    F_abl = F.copy()    # F is a numpy array (build_triple returns .numpy())
    F_abl[:, cols] = 0.0
    return (C, F_abl, mass, meta)


# ---------------------------------------------------------------------------
# Worker: per-process caches for DAGs, classifications, and BASE triples.
# Triple cache is LRU-bounded so memory stays predictable on the cluster.
# ---------------------------------------------------------------------------
_WORKER_DAGS: dict[tuple[str, str], dict] = {}
_WORKER_CLASSES: dict[tuple[str, str], dict] = {}
_WORKER_TRIPLES: OrderedDict = OrderedDict()
_TRIPLE_CACHE_CAP = 3   # keep at most 3 base triples per worker.
                        # Lowered from 6 after OOM cascade on piora-3GB-per-cpu:
                        # Qwen3-235B's base triple is ~600 MB, so 6 cached triples
                        # = 3.6 GB JUST IN CACHE. 3 entries (= source + target +
                        # one staging slot) is enough for paired-job locality.


def _get_dag(model: str, task: str, result_path: str) -> dict:
    key = (model, task)
    if key not in _WORKER_DAGS:
        p = Path(result_path) / "circuits" / f"dag_{model}_{task}.pt"
        _WORKER_DAGS[key] = torch.load(p, weights_only=False)
    return _WORKER_DAGS[key]


def _get_classification(model: str, task: str, result_path: str) -> dict:
    key = (model, task)
    if key not in _WORKER_CLASSES:
        p = Path(result_path) / "circuits" / CACHE_DIR_NAME / f"classify_{model}_{task}.pkl"
        with open(p, "rb") as f:
            _WORKER_CLASSES[key] = pickle.load(f)
    return _WORKER_CLASSES[key]


def _get_base_triple(model: str, task: str, q_idx: int, result_path: str):
    """Get the BASE (un-ablated) triple for (model, task, Q). LRU-cached."""
    key = (model, task, q_idx)
    if key in _WORKER_TRIPLES:
        _WORKER_TRIPLES.move_to_end(key)
        return _WORKER_TRIPLES[key]
    dag = _get_dag(model, task, result_path)
    cls = _get_classification(model, task, result_path)
    Q = QUANTILES[q_idx]
    theta = q_threshold(dag, Q)
    triple, _ = _build_filtered_triple(dag, cls, theta,
                                       act_norm_method=_ACT_NORM_METHOD,
                                       load_norm_method=_LOAD_NORM_METHOD)
    _WORKER_TRIPLES[key] = triple
    while len(_WORKER_TRIPLES) > _TRIPLE_CACHE_CAP:
        _WORKER_TRIPLES.popitem(last=False)
    return triple


def _worker_compute(args: tuple[int, int, int, int, int, str]) -> dict:
    """One FGW call: (mi, mj, t_idx, abl_idx, q_idx, result_path) -> S."""
    mi, mj, t_idx, abl_idx, q_idx, result_path = args
    model_i = MODELS[mi]
    model_j = MODELS[mj]
    task = TASKS[t_idx]
    abl = ABLATION_NAMES[abl_idx]

    base_i = _get_base_triple(model_i, task, q_idx, result_path)
    base_j = _get_base_triple(model_j, task, q_idx, result_path)

    t_i = _apply_ablation(base_i, abl)
    t_j = _apply_ablation(base_j, abl)

    d, _ = fgw_distance(t_i, t_j, alpha=ALPHA, n_init=N_INIT, seed=0)
    S = float(np.exp(-d))
    return {"mi": mi, "mj": mj, "t": t_idx, "abl": abl_idx, "q": q_idx,
            "d": float(d), "S": S}


# ---------------------------------------------------------------------------
# Per-pair-task processing (SLURM-array-friendly path, NOT pool-based).
#
# One SLURM array task per (model_pair, dataset) tuple. Each task:
#   1. Loads the 2 DAGs + 2 classifications.
#   2. Builds the 6 base triples (2 models x 3 Q values) sequentially.
#   3. Iterates 7 ablations x 3 Q values = 21 FGW calls sequentially.
#   4. Saves a (7, 3) tensor + (mi, mj, t_idx) metadata.
#
# Total: 28 model-pairs x 8 tasks = 224 SLURM array tasks. Each is fully
# independent: no ProcessPool to break, no cascading failures. If one task
# OOMs on Qwen3-235B x Qwen3-235B it's an isolated loss, not catastrophic.
# ---------------------------------------------------------------------------
def _pair_idx_to_mij(pair_idx: int, n_m: int = 8) -> tuple[int, int]:
    """Map pair_idx in [0, C(n_m, 2)) to (mi, mj) with mi < mj.
    Lexicographic order: (0,1), (0,2), ..., (0, n_m-1), (1,2), ..."""
    pairs = [(i, j) for i in range(n_m) for j in range(i + 1, n_m)]
    return pairs[pair_idx]


def _process_one_pair_task(pair_task_idx: int, result_path: str, out_dir: Path) -> None:
    """Sequentially compute (7 ablations x 3 Q values) for one (m_i, m_j, task)."""
    n_t = len(TASKS)
    n_pairs = len(MODELS) * (len(MODELS) - 1) // 2
    n_total = n_pairs * n_t
    if pair_task_idx < 0 or pair_task_idx >= n_total:
        print(f"ERROR: --pair-task-idx={pair_task_idx} not in [0, {n_total})")
        sys.exit(1)

    pair_idx = pair_task_idx // n_t
    t_idx = pair_task_idx % n_t
    mi, mj = _pair_idx_to_mij(pair_idx, n_m=len(MODELS))
    model_i, model_j = MODELS[mi], MODELS[mj]
    task = TASKS[t_idx]

    out_npz = out_dir / f"S_loo_pair_{pair_task_idx:03d}.npz"
    print(f"Pair-task    : idx={pair_task_idx}  pair={mi},{mj}  task_idx={t_idx}")
    print(f"  src        : {model_i} / {task}")
    print(f"  tgt        : {model_j} / {task}")
    print(f"  Output     : {out_npz}")

    # Skip if the output already exists and is non-empty (resume-friendly).
    if out_npz.exists() and out_npz.stat().st_size > 0:
        d = np.load(out_npz, allow_pickle=True)
        n_done = int((~np.isnan(d["S"])).sum())
        if n_done == len(ABLATION_NAMES) * len(QUANTILES):
            print(f"  SKIP: output already complete ({n_done} cells)\n")
            return
        else:
            print(f"  output is partial ({n_done} cells); recomputing\n")
    else:
        print()

    t0 = time.time()
    # Load DAGs + classifications.
    dag_i = torch.load(Path(result_path) / "circuits" / f"dag_{model_i}_{task}.pt",
                       weights_only=False)
    dag_j = torch.load(Path(result_path) / "circuits" / f"dag_{model_j}_{task}.pt",
                       weights_only=False)
    with open(Path(result_path) / "circuits" / CACHE_DIR_NAME /
              f"classify_{model_i}_{task}.pkl", "rb") as f:
        cls_i = pickle.load(f)
    with open(Path(result_path) / "circuits" / CACHE_DIR_NAME /
              f"classify_{model_j}_{task}.pkl", "rb") as f:
        cls_j = pickle.load(f)
    print(f"  loaded DAGs + classifications ({time.time() - t0:.1f}s)", flush=True)

    # Build 6 base triples (2 models x 3 Q values).
    triples_i: dict[int, tuple] = {}
    triples_j: dict[int, tuple] = {}
    for q_idx, Q in enumerate(QUANTILES):
        theta_i = q_threshold(dag_i, Q)
        triples_i[q_idx], n_keep_i = _build_filtered_triple(
            dag_i, cls_i, theta_i,
            act_norm_method=_ACT_NORM_METHOD,
            load_norm_method=_LOAD_NORM_METHOD)
        theta_j = q_threshold(dag_j, Q)
        triples_j[q_idx], n_keep_j = _build_filtered_triple(
            dag_j, cls_j, theta_j,
            act_norm_method=_ACT_NORM_METHOD,
            load_norm_method=_LOAD_NORM_METHOD)
        print(f"  built triples at Q={Q}: |V|_i={n_keep_i}, |V|_j={n_keep_j}  "
              f"({time.time() - t0:.1f}s)", flush=True)

    # Free DAGs + classifications -- we only need the triples from here on.
    del dag_i, dag_j, cls_i, cls_j

    # Iterate ablations x Q.
    n_abl = len(ABLATION_NAMES)
    n_q = len(QUANTILES)
    S_one = np.full((n_abl, n_q), np.nan, dtype=np.float64)
    n_failed = 0

    for abl_idx, abl in enumerate(ABLATION_NAMES):
        for q_idx, Q in enumerate(QUANTILES):
            try:
                t_i = _apply_ablation(triples_i[q_idx], abl)
                t_j = _apply_ablation(triples_j[q_idx], abl)
                d, _ = fgw_distance(t_i, t_j, alpha=ALPHA, n_init=N_INIT, seed=0)
                S_one[abl_idx, q_idx] = float(np.exp(-d))
                print(f"  abl={abl:<10s}  Q={Q:>5.3f}  S={S_one[abl_idx, q_idx]:.4f}  "
                      f"({time.time() - t0:.1f}s)", flush=True)
            except Exception as e:
                n_failed += 1
                print(f"  FAIL abl={abl} Q={Q}: {type(e).__name__}: {str(e)[:200]}",
                      flush=True)

    np.savez(
        out_npz,
        S=S_one,
        mi=mi, mj=mj, t_idx=t_idx,
        model_i=model_i, model_j=model_j, task=task,
        ablations=np.array(ABLATION_NAMES, dtype=object),
        quantiles=np.array(QUANTILES),
        alpha=ALPHA, beta=BETA, n_init=N_INIT,
    )
    print(f"\nDone in {time.time() - t0:.1f}s.  Failed: {n_failed}/{n_abl * n_q}")
    print(f"Saved: {out_npz}")


# ---------------------------------------------------------------------------
# Main.
# ---------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--chunk", type=int, default=None,
                        help="Which slice of jobs to process (0..num_chunks-1). "
                             "Defaults to $SLURM_ARRAY_TASK_ID if set, else 0.")
    parser.add_argument("--num-chunks", type=int, default=1,
                        help="How many SLURM array tasks split the job list "
                             "across nodes. Default 1 (single-node run).")
    parser.add_argument("--merge", action="store_true",
                        help="Skip computation; merge all S_loo_chunk*.npz AND "
                             "S_loo_pair_*.npz files into S_loo.npz and run the "
                             "analysis. Use this after all SLURM array tasks finish.")
    parser.add_argument("--pair-task-idx", type=int, default=None,
                        help="Per-pair-task mode: process ONE (model_pair, task) "
                             "combination across all 7 ablations x 3 Q values "
                             "sequentially in a single process. Index in "
                             "[0, 224): pair_idx = idx // 8, t_idx = idx %% 8. "
                             "Picks up SLURM_ARRAY_TASK_ID if not given. "
                             "This mode is robust to single-job OOM (no shared "
                             "ProcessPool) -- preferred over --chunk for "
                             "production SLURM array runs.")
    parser.add_argument("--act-norm", type=str, default="rank",
                        choices=["rank", "log_max"],
                        help="Normalisation method for the activation feature "
                             "passed through to fgw.build_triple. "
                             "'rank' (default) reproduces the legacy behaviour. "
                             "'log_max' uses log(1+act) / log(1+max(act)).")
    parser.add_argument("--load-norm", type=str, default="raw",
                        choices=["raw", "log_max"],
                        help="Normalisation method for the load feature passed "
                             "through to fgw.build_triple. "
                             "'raw' (default) reproduces the legacy behaviour "
                             "(load = n_tok / mean_in_layer, range up to N). "
                             "'log_max' uses per-layer log(1+load) / "
                             "log(1+max_in_layer(load)), bounded in [0, 1]. "
                             "Output is routed to a normalisation-specific "
                             "subdirectory (e.g. .../feature_ablation_logact_logload/) "
                             "to preserve legacy results.")
    args = parser.parse_args()

    # Apply act- and load-normalisation choices to the module-level globals
    # used by the triple-builder. Workers in per-pair-task mode (each SLURM
    # array task is its own Python process) see these set; for chunk mode the
    # spawn workers re-import the module and use the default values -- so
    # chunk mode only works with the defaults. Use --pair-task-idx for
    # non-default runs.
    global _ACT_NORM_METHOD, _LOAD_NORM_METHOD
    _ACT_NORM_METHOD = args.act_norm
    _LOAD_NORM_METHOD = args.load_norm

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    result_path = cfg["result_path"]
    # Route output to a normalisation-specific directory so legacy results are
    # never overwritten. Suffix is built from the non-default normalisations.
    suffix_parts: list[str] = []
    if args.act_norm == "log_max":
        suffix_parts.append("logact")
    if args.load_norm == "log_max":
        suffix_parts.append("logload")
    if suffix_parts:
        out_name = "feature_ablation_" + "_".join(suffix_parts)
    else:
        out_name = "feature_ablation"
    out_dir = Path(result_path) / "circuits" / out_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_npz_final = out_dir / "S_loo.npz"
    out_json = out_dir / "loo_summary.json"

    # -- Per-pair-task mode (preferred for SLURM array runs) ---------------
    # If --pair-task-idx is set (explicitly or via SLURM_ARRAY_TASK_ID under
    # a no-num-chunks invocation), short-circuit: do ONE (pair, task) tuple
    # sequentially and exit. No ProcessPool, no cascade risk.
    if args.pair_task_idx is not None:
        _process_one_pair_task(args.pair_task_idx, result_path, out_dir)
        return

    # Chunking config: either explicit --chunk, or pick up SLURM_ARRAY_TASK_ID.
    if args.chunk is None:
        args.chunk = int(os.environ.get("SLURM_ARRAY_TASK_ID", 0))
    if args.num_chunks > 1 and args.chunk >= args.num_chunks:
        print(f"ERROR: --chunk={args.chunk} >= --num-chunks={args.num_chunks}")
        sys.exit(1)

    # Output path depends on whether we're chunking.
    if args.num_chunks > 1:
        out_npz = out_dir / f"S_loo_chunk{args.chunk}_of_{args.num_chunks}.npz"
    else:
        out_npz = out_npz_final

    n_workers = args.workers or int(os.environ.get("SLURM_CPUS_PER_TASK", 8))

    n_abl = len(ABLATION_NAMES)
    n_q   = len(QUANTILES)
    n_t   = len(TASKS)
    n_m   = len(MODELS)

    print(f"Project root : {ROOT}")
    print(f"Result path  : {result_path}")
    print(f"Output dir   : {out_dir}")
    print(f"Settings     : alpha={ALPHA}, beta={BETA}, n_init={N_INIT}")
    print(f"Models       : {n_m}, tasks: {n_t}, Q axis: {QUANTILES}")
    print(f"Ablations    : {ABLATION_NAMES}")
    print(f"Workers      : {n_workers}")

    # Build job list. For each task, every unordered model pair (i < j) x
    # every (ablation, Q). Submit in (task -> pair -> abl -> Q) order so a
    # worker's LRU triple cache stays warm across consecutive jobs for the
    # same pair.
    jobs: list = []
    for t_idx in range(n_t):
        for mi in range(n_m):
            for mj in range(mi + 1, n_m):
                for abl_idx in range(n_abl):
                    for q_idx in range(n_q):
                        jobs.append((mi, mj, t_idx, abl_idx, q_idx, result_path))
    n_jobs_total = len(jobs)
    n_pairs = n_m * (n_m - 1) // 2 * n_t
    print(f"Pairs        : {n_pairs} unordered same-task cross-model")
    print(f"Total jobs   : {n_jobs_total}  ({n_pairs} pairs x {n_abl} abl x {n_q} Q)")

    # S[abl, q, t, mi, mj] tensor; symmetric in (mi, mj); diagonal NaN.
    S = np.full((n_abl, n_q, n_t, n_m, n_m), np.nan, dtype=np.float64)

    # --- Merge-only path: combine partial results and run analysis, no compute. ---
    if args.merge:
        chunk_files = sorted(out_dir.glob("S_loo_chunk*_of_*.npz"))
        pair_files  = sorted(out_dir.glob("S_loo_pair_*.npz"))
        if not chunk_files and not pair_files:
            print(f"ERROR: --merge mode but no S_loo_chunk* or S_loo_pair_* files "
                  f"in {out_dir}")
            sys.exit(1)
        print(f"\nMerge mode: combining {len(chunk_files)} chunk file(s) "
              f"and {len(pair_files)} pair-task file(s) ...")
        # Pair-task files first (per-(pair, task), shape (n_abl, n_q)):
        for f in pair_files:
            d = np.load(f, allow_pickle=True)
            S_pair = d["S"]                    # (n_abl, n_q)
            mi = int(d["mi"]); mj = int(d["mj"]); t_idx = int(d["t_idx"])
            non_nan = ~np.isnan(S_pair)
            # Fill both (mi, mj) and (mj, mi) for symmetry.
            for abl_idx in range(n_abl):
                for q_idx in range(n_q):
                    if non_nan[abl_idx, q_idx]:
                        S[abl_idx, q_idx, t_idx, mi, mj] = S_pair[abl_idx, q_idx]
                        S[abl_idx, q_idx, t_idx, mj, mi] = S_pair[abl_idx, q_idx]
            n_filled = int(non_nan.sum())
            print(f"  {f.name}: filled {n_filled} cells (mi={mi}, mj={mj}, t={t_idx})")
        # Then chunk files (full-tensor shape, only some cells non-NaN):
        for f in chunk_files:
            d = np.load(f, allow_pickle=True)
            S_chunk = d["S"]
            non_nan = ~np.isnan(S_chunk)
            S[non_nan] = S_chunk[non_nan]
            print(f"  {f.name}: filled {int(non_nan.sum())} cells")
        # Save merged tensor.
        np.savez(
            out_npz_final, S=S,
            models=np.array(MODELS, dtype=object),
            tasks=np.array(TASKS, dtype=object),
            quantiles=np.array(QUANTILES),
            ablations=np.array(ABLATION_NAMES, dtype=object),
            alpha=ALPHA, beta=BETA, n_init=N_INIT,
        )
        print(f"Saved merged: {out_npz_final}")
        print(f"  Total NaN remaining: {int(np.isnan(S).sum())} / {S.size}\n")
        # Fall through to analysis below.
        # (Skip the compute block by faking a finished state.)
        completed = S.size - int(np.isnan(S).sum())
        failed = 0
        t0 = time.time()
        _skip_compute = True
    else:
        _skip_compute = False

    # --- Slice the job list for this chunk (single-node default: 1 chunk). ---
    if not _skip_compute:
        if args.num_chunks > 1:
            chunk_size = (n_jobs_total + args.num_chunks - 1) // args.num_chunks
            start = args.chunk * chunk_size
            end = min(start + chunk_size, n_jobs_total)
            jobs = jobs[start:end]
            print(f"Chunk        : {args.chunk}/{args.num_chunks} -- "
                  f"jobs [{start}..{end - 1}], {len(jobs)} jobs this task")
        print(f"Output       : {out_npz}\n")

    n_jobs = len(jobs)   # count for THIS chunk (or all jobs in single-node mode)

    def _save_checkpoint(tag: str = "") -> None:
        np.savez(
            out_npz,
            S=S,
            models=np.array(MODELS, dtype=object),
            tasks=np.array(TASKS, dtype=object),
            quantiles=np.array(QUANTILES),
            ablations=np.array(ABLATION_NAMES, dtype=object),
            alpha=ALPHA, beta=BETA, n_init=N_INIT,
        )
        if tag:
            print(f"    [checkpoint {tag}] saved {out_npz.name}  "
                  f"NaN={int(np.isnan(S).sum())}", flush=True)

    if not _skip_compute:
        t0 = time.time()
        completed = 0
        failed = 0
        last_print = 0.0
        last_save = 0.0
        save_interval = 60.0

        ctx = mp.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=n_workers, mp_context=ctx) as ex:
                futs = {ex.submit(_worker_compute, j): j for j in jobs}
                for f in as_completed(futs):
                    try:
                        r = f.result()
                        abl_idx, q_idx, t_idx = r["abl"], r["q"], r["t"]
                        mi, mj = r["mi"], r["mj"]
                        S[abl_idx, q_idx, t_idx, mi, mj] = r["S"]
                        S[abl_idx, q_idx, t_idx, mj, mi] = r["S"]
                    except Exception as e:
                        failed += 1
                        mi, mj, t_idx, abl_idx, q_idx, _ = futs[f]
                        if failed <= 20 or failed % 100 == 0:
                            print(f"  FAIL [{failed}]: {MODELS[mi]}x{MODELS[mj]}  "
                                  f"task={TASKS[t_idx]}  Q={QUANTILES[q_idx]}  "
                                  f"abl={ABLATION_NAMES[abl_idx]}  "
                                  f"reason={type(e).__name__}: {str(e)[:120]}",
                                  flush=True)
                    completed += 1
                    now = time.time()
                    if now - last_print > 5.0 or completed in (1, n_jobs):
                        last_print = now
                        print(f"  [{completed:>5d}/{n_jobs}]  "
                              f"ok={completed - failed}  fail={failed}  "
                              f"(t={now - t0:.0f}s)", flush=True)
                    if now - last_save > save_interval:
                        last_save = now
                        _save_checkpoint(tag=f"{completed}/{n_jobs}")
        except Exception as e:
            print(f"\n!! Pool died: {type(e).__name__}: {e}")
            print(f"   Salvaging {completed - failed} completed cells.")

        print(f"\nDone in {time.time() - t0:.0f}s.")
        print(f"  Completed: {completed - failed}/{n_jobs}")
        print(f"  Failed   : {failed}/{n_jobs}")
        print(f"  NaN      : {int(np.isnan(S).sum())} / {S.size}")
        _save_checkpoint()
        print(f"Saved: {out_npz}\n")

        # In multi-chunk mode this task is finished; the analysis happens after
        # a separate `--merge` invocation on the union of chunks.
        if args.num_chunks > 1:
            print("Multi-chunk mode: skipping analysis (run with --merge after "
                  "all chunks finish to aggregate and print the LOO table).")
            return

    # ----------------------------------------------------------------------
    # Analysis: within-family vs cross-family premium, per ablation, per Q.
    # ----------------------------------------------------------------------
    print("Aggregating within/cross-family premium per (ablation, Q):\n")
    summary: dict = {}
    print(f"{'ablation':<10s}  {'Q':>6s}  {'within':>8s}  {'cross':>8s}  "
          f"{'gap':>8s}  {'gap_vs_full':>12s}  "
          f"{'n_within':>9s}  {'n_cross':>8s}")
    print("-" * 84)
    # Reference (full) gap, for the delta column.
    full_gap = {qi: None for qi in range(n_q)}
    for abl_idx, abl in enumerate(ABLATION_NAMES):
        for q_idx, Q in enumerate(QUANTILES):
            within_vals = []
            cross_vals = []
            for t_idx in range(n_t):
                for mi in range(n_m):
                    for mj in range(mi + 1, n_m):
                        s = S[abl_idx, q_idx, t_idx, mi, mj]
                        if np.isnan(s):
                            continue
                        fam_i = MODEL_TO_FAMILY.get(MODELS[mi])
                        fam_j = MODEL_TO_FAMILY.get(MODELS[mj])
                        same_fam = (fam_i is not None and fam_i == fam_j)
                        (within_vals if same_fam else cross_vals).append(s)
            within = float(np.mean(within_vals)) if within_vals else float("nan")
            cross  = float(np.mean(cross_vals))  if cross_vals  else float("nan")
            gap    = within - cross
            if abl == "full":
                full_gap[q_idx] = gap
            delta_full = (gap - full_gap[q_idx]) if full_gap[q_idx] is not None else float("nan")
            summary.setdefault(abl, {})[f"Q={Q}"] = {
                "within_mean": within, "cross_mean": cross, "gap": gap,
                "gap_vs_full": delta_full,
                "n_within": len(within_vals), "n_cross": len(cross_vals),
            }
            print(f"{abl:<10s}  {Q:>6.3f}  {within:>8.4f}  {cross:>8.4f}  "
                  f"{gap:>+8.4f}  {delta_full:>+12.4f}  "
                  f"{len(within_vals):>9d}  {len(cross_vals):>8d}")
        print()

    with open(out_json, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved: {out_json}")
    print(f"\nTotal wallclock: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
