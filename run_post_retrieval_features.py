"""
run_post_retrieval_features.py

Generates features_{enc}_v3.json for every encoder x dataset combination.
Each v3 file contains the original 17 query features merged with 12 new
post-retrieval features (29 total), keeping the same oracle targets from v2.

Strategy:
  - BM25 is encoder-agnostic, so it is run ONCE per dataset and shared.
  - Dense retrieval reuses cached document embeddings from results/cache/.
  - For all encoders, the existing v2 features file is loaded and the 12
    new features are appended; this avoids re-running original extraction.

    Encoder       features_in              features_out
    --------      -----------------------  -----------------------
    minilm        features_minilm_v2.json  features_minilm_v3.json
    bge           features_bge_v2.json     features_bge_v3.json
    e5            features_e5.json         features_e5_v3.json

Output schema (identical to v2 but num_features=29 and version="v3"):
  {
    "metadata": { "num_features": 29, "version": "v3", ... },
    "queries": {
      "<qid>": {
        "features": { <29 feature keys>: float },
        "oracle_alpha": float,
        "oracle_ndcg":  float,
        "is_informative": bool
      }
    }
  }

Usage:
    python run_post_retrieval_features.py
    python run_post_retrieval_features.py --encoders minilm bge
    python run_post_retrieval_features.py --datasets scifact arguana
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from src.config import DATASETS
from src.data.download_beir import download_beir_dataset
from src.data.load_beir import load_beir_dataset
from src.retrieval.bm25_retriever import run_bm25
from src.retrieval.dense_retriever import run_dense_retrieval
from src.features.query_features import FEATURE_NAMES
from src.features.post_retrieval_features import (
    NEW_FEATURE_NAMES,
    extract_post_retrieval_features,
)

RESULTS_DIR = Path("results")

ALL_FEATURE_NAMES = FEATURE_NAMES + NEW_FEATURE_NAMES  # 17 + 12 = 29

ENCODER_CONFIGS = {
    "minilm": {
        "model":        "all-MiniLM-L6-v2",
        "cache_dir":    "results/cache/dense",
        "features_in":  "features_minilm_v2.json",
        "features_out": "features_minilm_v3.json",
        "label":        "MiniLM",
    },
    "bge": {
        "model":        "BAAI/bge-base-en-v1.5",
        "cache_dir":    "results/cache/dense_bge",
        "features_in":  "features_bge_v2.json",
        "features_out": "features_bge_v3.json",
        "label":        "BGE",
    },
    "e5": {
        "model":        "intfloat/e5-base-v2",
        "cache_dir":    "results/cache/dense_e5",
        "features_in":  "features_e5.json",   # v2-format file from run_e5_pipeline.py
        "features_out": "features_e5_v3.json",
        "label":        "E5",
    },
}


# ── Per-encoder x dataset processing ─────────────────────────────────────────

def compute_v3_for_encoder(
    enc_key: str,
    cfg: dict,
    ds: str,
    bm25_results: dict,
    dense_results: dict,
) -> dict:
    """
    Load v2 features, compute 12 new features, merge, return v3 data dict.
    bm25_results and dense_results are pre-computed for this dataset.
    """
    features_in_path = RESULTS_DIR / ds / cfg["features_in"]
    with open(features_in_path, encoding="utf-8") as f:
        v2_data = json.load(f)

    v2_queries = v2_data["queries"]
    n_warn = 0
    query_data = {}

    for qid, v2_entry in v2_queries.items():
        bm25_q  = bm25_results.get(qid, {})
        dense_q = dense_results.get(qid, {})

        try:
            new_feats = extract_post_retrieval_features(bm25_q, dense_q)
        except Exception as exc:
            if n_warn < 5:
                print(f"    [WARN] qid={qid}: {exc}")
            n_warn += 1
            new_feats = {k: 0.0 for k in NEW_FEATURE_NAMES}

        merged = {**v2_entry["features"], **new_feats}

        query_data[qid] = {
            "features":       merged,
            "oracle_alpha":   v2_entry["oracle_alpha"],
            "oracle_ndcg":    v2_entry["oracle_ndcg"],
            "is_informative": v2_entry["is_informative"],
        }

    if n_warn > 0:
        print(f"    [WARN] {n_warn} queries used 0.0 fallback for new features")

    n_inf = sum(1 for d in query_data.values() if d["is_informative"])
    return {
        "metadata": {
            "dataset":       ds,
            "encoder":       enc_key,
            "model":         cfg["model"],
            "num_queries":   len(query_data),
            "num_informative": n_inf,
            "num_features":  len(ALL_FEATURE_NAMES),
            "feature_names": ALL_FEATURE_NAMES,
            "version":       "v3",
            "timestamp":     datetime.now(timezone.utc).isoformat(),
        },
        "queries": query_data,
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate v3 features (17 original + 12 new post-retrieval = 29 total)."
    )
    parser.add_argument(
        "--encoders", nargs="+", default=["minilm", "bge", "e5"],
        choices=["minilm", "bge", "e5"],
        help="Which encoders to process (default: all three)",
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("POST-RETRIEVAL FEATURE GENERATION  (v3)")
    print(f"Encoders : {args.encoders}")
    print(f"Datasets : {args.datasets}")
    print(f"New features   : {len(NEW_FEATURE_NAMES)}  -> {NEW_FEATURE_NAMES}")
    print(f"Total features : {len(ALL_FEATURE_NAMES)}")
    print("=" * 70)

    t_wall = time.perf_counter()
    generated = []

    for ds in args.datasets:
        print(f"\n{'=' * 60}")
        print(f"Dataset: {ds}")
        print("=" * 60)

        download_beir_dataset(data_dir=args.data_dir, dataset_name=ds)
        corpus, queries, _ = load_beir_dataset(
            data_dir=args.data_dir, dataset_name=ds, split=args.split
        )
        print(f"  corpus={len(corpus):,}  queries={len(queries):,}")

        # BM25 is encoder-agnostic: run once per dataset
        t0 = time.perf_counter()
        bm25_results = run_bm25(corpus, queries, top_k=args.top_k)
        print(f"  BM25 done ({time.perf_counter() - t0:.1f}s)")

        for enc_key in args.encoders:
            cfg = ENCODER_CONFIGS[enc_key]
            features_in_path = RESULTS_DIR / ds / cfg["features_in"]
            if not features_in_path.exists():
                print(f"  [{enc_key}] SKIP -- {features_in_path} not found")
                continue

            print(f"\n  [{cfg['label']}] dense retrieval...")
            t0 = time.perf_counter()
            cache_key = f"{ds}::{args.split}::{cfg['model']}"
            dense_results = run_dense_retrieval(
                corpus, queries,
                model_name=cfg["model"],
                top_k=args.top_k,
                cache_dir=cfg["cache_dir"],
                cache_key=cache_key,
            )
            print(f"  [{cfg['label']}] dense done ({time.perf_counter() - t0:.1f}s)")

            t0 = time.perf_counter()
            v3_data = compute_v3_for_encoder(enc_key, cfg, ds, bm25_results, dense_results)
            n_q   = v3_data["metadata"]["num_queries"]
            n_inf = v3_data["metadata"]["num_informative"]

            out_path = RESULTS_DIR / ds / cfg["features_out"]
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(v3_data, f, indent=2)

            print(
                f"  [{cfg['label']}] saved -> {out_path}  "
                f"queries={n_q}  informative={n_inf}  "
                f"({time.perf_counter() - t0:.1f}s)"
            )
            generated.append(str(out_path))

    elapsed = time.perf_counter() - t_wall
    print(f"\n{'=' * 70}")
    print(f"Done in {elapsed:.0f}s.  Generated {len(generated)} files:")
    for p in generated:
        print(f"  {p}")
    print()
    print("Next step:")
    print("  python run_phase1_experiments.py")


if __name__ == "__main__":
    main()
