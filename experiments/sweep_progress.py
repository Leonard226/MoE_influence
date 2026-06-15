"""Progress / ETA / health monitor for the headline α × Q sweep.

Reads all sweep_src*_chunk*.npz slice files under
    ${result_path}/circuits/alpha_beta_sweep_logact_logload/
and reports:
  - overall completion % (cells filled / total)
  - per-task progress and rolling ETA (n_remaining / observed_rate)
  - overall ETA = max ETA across in-progress tasks (they run in parallel)
  - health flags: STALE (file not touched recently), FAILED (S contains -1),
    NOT_STARTED (no file yet)

Usage:
    python experiments/sweep_progress.py
    python experiments/sweep_progress.py --subdir alpha_beta_sweep         # legacy
    python experiments/sweep_progress.py --stale-min 15 --verbose

Per-task time accounting: ctime of the .npz is used as task-start proxy
(the file is created at the first per-Q checkpoint, ~30-60s into the task).
This biases the rate slightly low for fast tasks; for the bottleneck Qwen3
sources the offset is negligible.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from experiments.run_alpha_beta_sweep import TUPLES, N_TUPLES, ALPHAS, QUANTILES

# Default headline-sweep layout: 64 sources × 2 target chunks.
DEFAULT_NUM_CHUNKS = 2


def _fmt_dur(s: float) -> str:
    if s < 0 or not np.isfinite(s):
        return "  --  "
    if s < 60:
        return f"{int(s):>4d}s"
    if s < 3600:
        return f"{int(s) // 60:>3d}m{int(s) % 60:02d}s"
    h = int(s) // 3600
    m = (int(s) % 3600) // 60
    return f"{h:>2d}h{m:02d}m"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--subdir", default="alpha_beta_sweep_logact_logload",
                        help="Subdir under {result_path}/circuits/ to scan.")
    parser.add_argument("--num-chunks", type=int, default=DEFAULT_NUM_CHUNKS,
                        help=f"Target-chunk count per source. Default {DEFAULT_NUM_CHUNKS}.")
    parser.add_argument("--stale-min", type=float, default=15.0,
                        help="Flag a task STALE if its file hasn't been touched "
                             "in this many minutes (default 15).")
    parser.add_argument("--verbose", action="store_true",
                        help="Print one row per slice file (including DONE).")
    args = parser.parse_args()

    with open(ROOT / "config.yaml") as f:
        cfg = yaml.safe_load(f)
    scan_dir = Path(cfg["result_path"]) / "circuits" / args.subdir
    if not scan_dir.is_dir():
        print(f"ERROR: directory not found: {scan_dir}")
        sys.exit(1)

    # ---- Expected slice topology ---------------------------------------
    # For src S with chunk C of K chunks, n_targets = depends on chunking
    # algorithm in run_alpha_beta_sweep.py:
    #   all_other = [0..63] \ {src}                       # 63 entries
    #   chunk_size = ceil(63 / K)
    #   chunk c -> all_other[c*chunk_size : (c+1)*chunk_size]
    n_a, n_q = len(ALPHAS), len(QUANTILES)
    expected: dict[tuple[int, int], int] = {}
    for src in range(N_TUPLES):
        all_other = [i for i in range(N_TUPLES) if i != src]
        chunk_size = (len(all_other) + args.num_chunks - 1) // args.num_chunks
        for c in range(args.num_chunks):
            n_tgt = len(all_other[c * chunk_size : (c + 1) * chunk_size])
            expected[(src, c)] = n_tgt
    n_total_cells = sum(n_tgt * n_a * n_q for n_tgt in expected.values())
    n_total_slices = len(expected)

    # ---- Walk slice files ----------------------------------------------
    pat = re.compile(r"sweep_src(\d+)_chunk(\d+)\.npz$")
    now = time.time()
    tasks: list[dict] = []   # one per existing file
    n_filled = 0
    n_failed_cells = 0
    found_keys: set[tuple[int, int]] = set()

    for f in sorted(scan_dir.glob("sweep_src*_chunk*.npz")):
        m = pat.search(f.name)
        if not m:
            continue
        src, chunk = int(m.group(1)), int(m.group(2))
        try:
            S = np.load(f, allow_pickle=True)["S"]
        except Exception as e:
            print(f"  [WARN] {f.name}: could not load ({e})")
            continue
        n_cells = S.size
        non_nan_mask = ~np.isnan(S)
        n_done_cells = int(non_nan_mask.sum())
        n_fail_cells = int(((S < 0) & non_nan_mask).sum())
        # Per-target completion: a target row is "done" if all its α×Q cells
        # are non-NaN (matches the per-Q checkpoint write semantics).
        per_tgt_done = (~np.isnan(S)).reshape(S.shape[0], -1).all(axis=1)
        n_tgt_done = int(per_tgt_done.sum())
        n_tgt_total = S.shape[0]
        stat = f.stat()
        elapsed = now - stat.st_ctime
        idle = now - stat.st_mtime
        # Observed rate over the task lifetime (targets / second).
        rate = n_tgt_done / elapsed if elapsed > 0 and n_tgt_done > 0 else 0.0
        n_tgt_rem = n_tgt_total - n_tgt_done
        eta = n_tgt_rem / rate if rate > 0 else float("inf")
        complete = (n_tgt_done == n_tgt_total)
        tasks.append({
            "src": src, "chunk": chunk,
            "tuple": TUPLES[src],
            "n_tgt_done": n_tgt_done, "n_tgt_total": n_tgt_total,
            "rate": rate, "eta": eta, "elapsed": elapsed, "idle": idle,
            "n_done_cells": n_done_cells, "n_cells": n_cells,
            "n_fail_cells": n_fail_cells,
            "complete": complete,
        })
        n_filled += n_done_cells
        n_failed_cells += n_fail_cells
        found_keys.add((src, chunk))

    # ---- Summary ---------------------------------------------------------
    n_started = len(tasks)
    n_done = sum(t["complete"] for t in tasks)
    n_running = sum((not t["complete"]) and t["idle"] < args.stale_min * 60 for t in tasks)
    n_stale = sum((not t["complete"]) and t["idle"] >= args.stale_min * 60 for t in tasks)
    n_not_started = n_total_slices - n_started

    in_progress = [t for t in tasks if not t["complete"]]
    overall_eta = max((t["eta"] for t in in_progress
                       if np.isfinite(t["eta"])), default=0.0)

    pct_cells = 100 * n_filled / n_total_cells

    print()
    print(f"=== {args.subdir} progress ===")
    print(f"  slices            : {n_started}/{n_total_slices} touched  "
          f"({n_done} done, {n_running} running, {n_stale} stale, "
          f"{n_not_started} not started)")
    print(f"  cells             : {n_filled}/{n_total_cells}  ({pct_cells:5.1f}%)")
    print(f"  failed cells (-1) : {n_failed_cells}")
    if in_progress:
        print(f"  ETA (max in-prog) : {_fmt_dur(overall_eta)}   "
              f"({sum(np.isfinite(t['eta']) for t in in_progress)} of "
              f"{len(in_progress)} have a rate estimate)")
    else:
        print(f"  ETA               : DONE")
    print()

    # ---- Per-source rows (in-progress + stale + failed first) -----------
    rows = sorted(tasks, key=lambda t: (t["complete"], -t["eta"] if np.isfinite(t["eta"]) else -1e18))
    # Group: stale first, then in-progress (slowest first), then done.
    def _sort_key(t):
        if not t["complete"] and t["idle"] >= args.stale_min * 60:
            bucket = 0
        elif t["n_fail_cells"] > 0:
            bucket = 1
        elif not t["complete"]:
            bucket = 2
        else:
            bucket = 3
        eta = t["eta"] if np.isfinite(t["eta"]) else 1e18
        return (bucket, -eta)
    rows = sorted(tasks, key=_sort_key)

    print(f"{'src':>3s} {'ch':>3s}  {'model/task':<28s}  "
          f"{'targets':>10s}  {'rate':>9s}  {'elapsed':>8s}  {'idle':>7s}  "
          f"{'ETA':>7s}  {'flags':<24s}")
    print("-" * 116)
    for t in rows:
        if t["complete"] and not args.verbose:
            continue
        model, task = t["tuple"]
        prog = f"{t['n_tgt_done']:>3d}/{t['n_tgt_total']:<3d}"
        rate_str = f"{t['rate']*60:5.2f}/min" if t["rate"] > 0 else "  --   "
        flags = []
        if t["complete"]:
            flags.append("DONE")
        if t["n_fail_cells"] > 0:
            flags.append(f"FAIL({t['n_fail_cells']})")
        if (not t["complete"]) and t["idle"] >= args.stale_min * 60:
            flags.append("STALE")
        flag_s = " ".join(flags)
        eta_s = _fmt_dur(t["eta"]) if not t["complete"] else "  --  "
        print(f"{t['src']:>3d} {t['chunk']:>3d}  {model}/{task:<10s}  "
              f"{prog:>10s}  {rate_str:>9s}  "
              f"{_fmt_dur(t['elapsed']):>8s}  {_fmt_dur(t['idle']):>7s}  "
              f"{eta_s:>7s}  {flag_s:<24s}")

    # ---- Not-started list ------------------------------------------------
    if n_not_started > 0:
        missing = [(s, c) for s in range(N_TUPLES) for c in range(args.num_chunks)
                   if (s, c) not in found_keys]
        print()
        print(f"Not started ({n_not_started}):")
        # Show first 15 then ellipsis.
        for s, c in missing[:15]:
            m, t = TUPLES[s]
            print(f"  src={s:>2d}  chunk={c}  {m}/{t}")
        if len(missing) > 15:
            print(f"  ... and {len(missing) - 15} more")

    print()


if __name__ == "__main__":
    main()
