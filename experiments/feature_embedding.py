"""Canonical 2D-UMAP and 3D-PCA embeddings of the per-expert feature space.

Single source of truth for all downstream feature visualisations. Computes
ONE UMAP and ONE PCA per task on the FULL pooled feature matrix F_all
(no subsampling), caches to disk, and serves the same coordinates to every
visualisation script so all plots share the exact same projection.

This eliminates the "two UMAPs look different because they used different
subsamples / different RNGs" class of artefacts. Different colourings of
the same UMAP are now strictly comparable.

Cache file
    ${result_path}/circuits/feature_inspection/embedding_<task>.npz

Cache contents
    F_all          (V_all, D)   per-expert features (log-max load/act)
    model_idx      (V_all,)     model index per expert (into MODELS)
    models         (8,) object  list of canonical model names
    umap_2d        (V_all, 2)   UMAP coordinates (full data, deterministic)
    pca_3d         (V_all, 3)   first 3 principal components
    pca_explained  (3,)         per-component variance explained
    task           ()           the task string (e.g. "c4")
    umap_params    ()           dict-as-string, for audit
    pca_params     ()           dict-as-string, for audit

Public API
    load_or_compute_embedding(task, recompute=False) -> dict

If the cache exists and `recompute` is False, returns the cached embedding
in ~1s. Otherwise fits UMAP + PCA fresh (~1-2 min for ~32k experts), saves,
and returns. Pass `recompute=True` after changing the feature construction
(e.g. switching normalisations) to invalidate.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

with open(os.path.join(ROOT, "config.yaml")) as f:
    _config = yaml.safe_load(f)
CACHE_DIR = Path(_config["result_path"]) / "circuits" / "feature_inspection"

# Canonical hyperparams. Changing these WILL produce a different embedding;
# delete the cache (or pass recompute=True) after any change here.
UMAP_PARAMS = dict(n_components=2, random_state=0, n_neighbors=30,
                   min_dist=0.1, metric="euclidean")
PCA_PARAMS = dict(n_components=3)


def _cache_path(task: str) -> Path:
    return CACHE_DIR / f"embedding_{task}.npz"


def _fit(F_all: np.ndarray) -> dict:
    """Compute UMAP and PCA on the full feature matrix."""
    from sklearn.decomposition import PCA
    print(f"  fitting PCA on {len(F_all)} points (full data) ...", flush=True)
    t0 = time.time()
    pca = PCA(**PCA_PARAMS)
    pca_3d = pca.fit_transform(F_all)
    print(f"    PCA done in {time.time() - t0:.1f}s "
          f"(var explained: {pca.explained_variance_ratio_})", flush=True)

    try:
        from umap import UMAP
    except ImportError as e:
        raise RuntimeError(
            "umap-learn is required for the canonical embedding. "
            "Install with: pip install umap-learn"
        ) from e
    print(f"  fitting UMAP on {len(F_all)} points (full data) ...", flush=True)
    t0 = time.time()
    umap_2d = UMAP(**UMAP_PARAMS).fit_transform(F_all)
    print(f"    UMAP done in {time.time() - t0:.1f}s", flush=True)

    return {
        "umap_2d": umap_2d.astype(np.float32),
        "pca_3d": pca_3d.astype(np.float32),
        "pca_explained": pca.explained_variance_ratio_.astype(np.float32),
        "umap_params": json.dumps(UMAP_PARAMS),
        "pca_params": json.dumps(PCA_PARAMS),
    }


def load_or_compute_embedding(task: str, recompute: bool = False) -> dict:
    """Return canonical embedding dict for the chosen task.

    Loads from cache if present (~1s); otherwise computes from scratch (~1-2 min)
    and writes the cache. Pass recompute=True to force a fresh fit.
    """
    cache = _cache_path(task)
    if cache.exists() and not recompute:
        print(f"  loading cached embedding from {cache.name}", flush=True)
        npz = np.load(cache, allow_pickle=True)
        d = {k: npz[k] for k in npz.files}
        # Convert object array back to list of strings.
        d["models"] = [str(m) for m in d["models"].tolist()]
        d["task"] = str(d["task"])
        d["umap_params"] = json.loads(str(d["umap_params"]))
        d["pca_params"] = json.loads(str(d["pca_params"]))
        return d

    print(f"  no cache (or recompute requested); building embedding for "
          f"task={task!r} ...", flush=True)
    # Deferred import to avoid circular dependency with inspect_features.
    from experiments.inspect_features import _build_all
    F_all, model_idx, models = _build_all(task)
    fitted = _fit(F_all)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        cache,
        F_all=F_all.astype(np.float32),
        model_idx=model_idx.astype(np.int32),
        models=np.array(models, dtype=object),
        task=task,
        **fitted,
    )
    print(f"  saved embedding to {cache.name} "
          f"({cache.stat().st_size / 1e6:.1f} MB)", flush=True)

    return {
        "F_all": F_all,
        "model_idx": model_idx,
        "models": models,
        "task": task,
        "umap_2d": fitted["umap_2d"],
        "pca_3d": fitted["pca_3d"],
        "pca_explained": fitted["pca_explained"],
        "umap_params": UMAP_PARAMS,
        "pca_params": PCA_PARAMS,
    }
