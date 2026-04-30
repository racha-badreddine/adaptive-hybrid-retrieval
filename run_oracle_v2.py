"""
run_oracle_v2.py

Recompute oracle alphas for MiniLM and BGE using the new 0.05-step grid
(21 points: 0.00, 0.05, ..., 1.00) over the FULL query set (not just the
"informative" subset).  Regenerates features_v2 files with the corrected
oracle targets so that adaptive models can be retrained.

Encodings are NOT re-run -- cached embeddings from previous runs are reused.

Outputs (per dataset, per encoder):
    results/{ds}/oracle_alpha_minilm_v2.json
    results/{ds}/oracle_alpha_bge_v2.json
    results/{ds}/features_minilm_v2.json
    results/{ds}/features_bge_v2.json

Then run the adaptive pipeline with the v2 files:
    python run_adaptive_pipeline.py \\
        --oracle-filename oracle_alpha_minilm_v2.json \\
        --features-filename features_minilm_v2.json \\
        --dense-label minilm_v2

    python run_adaptive_pipeline.py \\
        --dense-model BAAI/bge-base-en-v1.5 \\
        --dense-cache-dir results/cache/dense_bge \\
        --oracle-filename oracle_alpha_bge_v2.json \\
        --features-filename features_bge_v2.json \\
        --dense-label bge_v2

Usage:
    python run_oracle_v2.py [--encoders minilm bge] [--datasets scifact ...]
"""

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.config import DATASETS, ALPHA_GRID
from src.data.download_beir import download_beir_dataset
from src.data.load_beir import load_beir_dataset
from src.retrieval.bm25_retriever import run_bm25
from src.retrieval.dense_retriever import run_dense_retrieval
from src.training.generate_oracle_alpha import generate_oracle_alphas, save_oracle_data
from src.features.query_features import (
    extract_query_features,
    compute_idf_dict,
    FEATURE_NAMES,
)

RESULTS_DIR = Path("results")
MINILM_MODEL = "all-MiniLM-L6-v2"
MINILM_CACHE_DIR = "results/cache/dense"
BGE_MODEL = "BAAI/bge-base-en-v1.5"
BGE_CACHE_DIR = "results/cache/dense_bge"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Recompute v2 oracle+features for MiniLM and BGE."
    )
    parser.add_argument(
        "--encoders", nargs="+", default=["minilm", "bge"],
        choices=["minilm", "bge"],
        help="Which encoders to process (default: both)",
    )
    parser.add_argument("--datasets", nargs="+", default=DATASETS)
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=100)
    return parser.parse_args()


def _oracle_stats(oracle_data: dict) -> dict:
    """Summary stats over ALL queries (not informative-only)."""
    all_entries = list(oracle_data.values())
    alphas = [d["best_alpha"] for d in all_entries]
    ndcgs  = [d["best_ndcg"]  for d in all_entries]
    n_inf  = sum(1 for d in all_entries if d.get("is_informative"))
    return {
        "n_total": len(all_entries),
        "n_informative": n_inf,
        "mean_alpha": float(np.mean(alphas)) if alphas else 0.0,
        "std_alpha":  float(np.std(alphas))  if alphas else 0.0,
        "mean_ndcg":  float(np.mean(ndcgs))  if ndcgs  else 0.0,
    }


def compute_oracle_and_features(
    ds: str,
    corpus: dict,
    queries: dict,
    qrels: dict,
    bm25_results: dict,
    dense_results: dict,
    top_k: int,
    oracle_out_filename: str,
    features_out_filename: str,
    encoder_label: str,
    model_name: str,
) -> dict:
    """
    Compute oracle over ALL queries with ALPHA_GRID (21-point 0.05 step),
    regenerate features JSON with updated oracle targets.
    Returns oracle_data dict.
    """
    query_ids = set(bm25_results.keys()) & set(dense_results.keys())
    qrels_ev = {q: qrels[q] for q in query_ids if q in qrels}

    print(f"  [{encoder_label}] Oracle grid search: {len(ALPHA_GRID)} alphas x "
          f"{len(qrels_ev)} queries ...")
    t0 = time.perf_counter()

    oracle_data = generate_oracle_alphas(
        qrels=qrels_ev,
        bm25_results=bm25_results,
        dense_results=dense_results,
        alpha_grid=ALPHA_GRID,
        top_k=top_k,
    )
    elapsed = time.perf_counter() - t0

    out_path = RESULTS_DIR / ds / oracle_out_filename
    save_oracle_data(oracle_data, out_path)

    stats = _oracle_stats(oracle_data)
    print(f"  [{encoder_label}] Oracle done in {elapsed:.1f}s")
    print(f"    mean_alpha={stats['mean_alpha']:.3f}  std={stats['std_alpha']:.3f}  "
          f"NDCG@10={stats['mean_ndcg']:.4f}  "
          f"informative={stats['n_informative']}/{stats['n_total']}")

    # Sanity: oracle NDCG with new 0.05 grid vs old oracle (if exists)
    old_key_map = {"minilm": "oracle_alpha.json", "bge": "oracle_alpha_bge.json"}
    old_path = RESULTS_DIR / ds / old_key_map.get(encoder_label, "")
    if old_path and old_path.exists():
        with open(old_path, encoding="utf-8") as f:
            old_oracle = json.load(f)
        old_ndcgs = [d["best_ndcg"] for d in old_oracle.values()]
        old_mean = float(np.mean(old_ndcgs)) if old_ndcgs else 0.0
        delta = stats["mean_ndcg"] - old_mean
        flag = " [FLAG: regression vs old oracle!]" if delta < -1e-6 else ""
        print(f"    vs old oracle (step~0.02): new={stats['mean_ndcg']:.4f}  "
              f"old={old_mean:.4f}  delta={delta:+.4f}{flag}")

    # Regenerate features with new oracle targets
    idf_dict = compute_idf_dict(corpus)
    query_data = {}
    for qid, query_text in queries.items():
        bm25_q   = bm25_results.get(qid, {})
        dense_q  = dense_results.get(qid, {})
        if not bm25_q and not dense_q:
            continue
        feats = extract_query_features(
            query_text=query_text,
            bm25_results_for_query=bm25_q,
            dense_results_for_query=dense_q,
            idf_dict=idf_dict,
        )
        oracle_entry = oracle_data.get(qid, {})
        query_data[qid] = {
            "features": feats,
            "oracle_alpha": oracle_entry.get("best_alpha"),
            "oracle_ndcg":  oracle_entry.get("best_ndcg"),
            "is_informative": oracle_entry.get("is_informative", False),
        }

    n_inf = sum(1 for d in query_data.values() if d["is_informative"])
    features_out = {
        "metadata": {
            "dataset": ds,
            "num_queries": len(query_data),
            "num_informative": n_inf,
            "num_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "dense_model": model_name,
            "alpha_grid": "0.05-step (21 points)",
            "oracle_version": "v2",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "queries": query_data,
    }

    feat_path = RESULTS_DIR / ds / features_out_filename
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(features_out, f, indent=2)
    print(f"  [{encoder_label}] Features saved -> {feat_path}  "
          f"(queries={len(query_data)}, informative={n_inf})")

    return oracle_data


def main():
    args = parse_args()
    print(f"\nORACLE V2 RECOMPUTATION")
    print(f"Encoders: {args.encoders}")
    print(f"Datasets: {args.datasets}")
    print(f"Alpha grid: {len(ALPHA_GRID)} points, step 0.05 (0.00..1.00)")
    print("=" * 70)

    encoder_configs = {
        "minilm": {
            "model": MINILM_MODEL,
            "cache_dir": MINILM_CACHE_DIR,
            "oracle_out": "oracle_alpha_minilm_v2.json",
            "features_out": "features_minilm_v2.json",
        },
        "bge": {
            "model": BGE_MODEL,
            "cache_dir": BGE_CACHE_DIR,
            "oracle_out": "oracle_alpha_bge_v2.json",
            "features_out": "features_bge_v2.json",
        },
    }

    for ds in args.datasets:
        print(f"\n{'=' * 70}")
        print(f"Dataset: {ds}")
        print("=" * 70)

        download_beir_dataset(data_dir=args.data_dir, dataset_name=ds)
        corpus, queries, qrels = load_beir_dataset(
            data_dir=args.data_dir, dataset_name=ds, split=args.split
        )
        print(f"  corpus={len(corpus):,}  queries={len(queries):,}")

        # BM25 (fast; no cache needed)
        t0 = time.perf_counter()
        bm25_results = run_bm25(corpus, queries, top_k=args.top_k)
        print(f"  BM25 done ({time.perf_counter() - t0:.1f}s)")

        for enc_name in args.encoders:
            cfg = encoder_configs[enc_name]
            print(f"\n  --- Encoder: {enc_name} ({cfg['model']}) ---")

            cache_key = f"{ds}::{args.split}::{cfg['model']}"
            t0 = time.perf_counter()
            dense_results = run_dense_retrieval(
                corpus, queries,
                model_name=cfg["model"],
                top_k=args.top_k,
                cache_dir=cfg["cache_dir"],
                cache_key=cache_key,
            )
            print(f"  Dense results loaded/computed ({time.perf_counter() - t0:.1f}s)")

            compute_oracle_and_features(
                ds=ds,
                corpus=corpus,
                queries=queries,
                qrels=qrels,
                bm25_results=bm25_results,
                dense_results=dense_results,
                top_k=args.top_k,
                oracle_out_filename=cfg["oracle_out"],
                features_out_filename=cfg["features_out"],
                encoder_label=enc_name,
                model_name=cfg["model"],
            )

    print(f"\n{'=' * 70}")
    print("ORACLE V2 COMPLETE")
    print("=" * 70)
    print("Generated files:")
    for ds in args.datasets:
        for enc_name in args.encoders:
            cfg = encoder_configs[enc_name]
            print(f"  results/{ds}/{cfg['oracle_out']}")
            print(f"  results/{ds}/{cfg['features_out']}")
    print()
    print("Next steps -- retrain adaptive models with v2 oracle targets:")
    for enc_name in args.encoders:
        cfg = encoder_configs[enc_name]
        label = f"{enc_name}_v2"
        model_flag = (f" --dense-model {cfg['model']} "
                      f"--dense-cache-dir {cfg['cache_dir']}" if enc_name != "minilm" else "")
        print(f"\n  python run_adaptive_pipeline.py{model_flag} \\")
        print(f"      --oracle-filename {cfg['oracle_out']} \\")
        print(f"      --features-filename {cfg['features_out']} \\")
        print(f"      --dense-label {label}")


if __name__ == "__main__":
    main()
