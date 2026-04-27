"""
run_adaptive_pipeline.py

Top-level script that orchestrates Phase 4:
  1. Load retrieval results for all processed datasets
  2. Run prepare_data + split_data + train_adaptive
  3. Run full evaluation (per-dataset, significance, transfer, efficiency)
  4. Print complete results table

Usage:
    python run_adaptive_pipeline.py [--datasets scifact arguana ...] [options]
"""

import argparse
import json
import time
from pathlib import Path

from src.data.download_beir import download_beir_dataset
from src.data.load_beir import load_beir_dataset
from src.retrieval.bm25_retriever import run_bm25
from src.retrieval.dense_retriever import run_dense_retrieval
from src.evaluation.evaluate_adaptive import run_full_evaluation

RESULTS_DIR = Path("results")


def parse_args():
    parser = argparse.ArgumentParser(description="Run adaptive alpha-prediction pipeline.")
    parser.add_argument(
        "--datasets", nargs="+", default=None,
        help="Datasets to include (default: all with features.json in results/)",
    )
    parser.add_argument("--data-dir", default="data")
    parser.add_argument("--split", default="test")
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--dense-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--dense-cache-dir", default="results/cache/dense")
    parser.add_argument(
        "--dense-label", default=None,
        help=(
            "Short label used in output filenames (e.g. 'bge' produces "
            "adaptive_model_results_bge.json and split suffix _bge). "
            "Auto-derived from --dense-model when omitted."
        ),
    )
    parser.add_argument(
        "--features-filename", default="features.json",
        help="Per-dataset features file to load for training (default: features.json)",
    )
    parser.add_argument(
        "--oracle-filename", default="oracle_alpha.json",
        help="Per-dataset oracle alpha file (default: oracle_alpha.json)",
    )
    return parser.parse_args()


def _derive_label(model_name: str) -> str:
    """Derive a short label from the model name for file naming."""
    name = model_name.lower()
    if "bge" in name:
        return "bge"
    if "minilm" in name or "miniLM" in model_name:
        return ""  # default MiniLM run — no suffix
    # Generic fallback: strip special chars
    import re
    return re.sub(r"[^a-z0-9]", "", name)[:12]


def main():
    args = parse_args()

    # Derive dense label for output naming
    dense_label = args.dense_label
    if dense_label is None:
        dense_label = _derive_label(args.dense_model)

    split_suffix = f"_{dense_label}" if dense_label else ""
    output_filename = (
        f"adaptive_model_results_{dense_label}.json"
        if dense_label
        else "adaptive_model_results.json"
    )

    # Discover datasets automatically if not specified
    if args.datasets is None:
        args.datasets = sorted(
            p.parent.name for p in RESULTS_DIR.glob(f"*/{args.features_filename}")
        )

    if not args.datasets:
        print(
            f"No datasets with {args.features_filename} found. "
            "Run the retrieval/oracle/feature pipeline first."
        )
        return

    print(f"\nDatasets: {args.datasets}")
    print(f"Dense model: {args.dense_model}")
    print(f"Dense label: '{dense_label}'  split_suffix='{split_suffix}'")
    print(f"Output: {output_filename}")
    print("=" * 70)

    all_retrieval_data = {}
    best_static_by_dataset = {}

    for ds in args.datasets:
        t0 = time.perf_counter()
        print(f"\n--- Loading {ds} ---")

        download_beir_dataset(data_dir=args.data_dir, dataset_name=ds)

        corpus, queries, qrels = load_beir_dataset(
            data_dir=args.data_dir,
            dataset_name=ds,
            split=args.split,
        )
        print(f"  corpus={len(corpus):,}  queries={len(queries):,}")

        # BM25
        bm25_results = run_bm25(corpus, queries, top_k=args.top_k)
        evaluated_qids = set(bm25_results.keys())
        qrels_eval = {q: qrels[q] for q in evaluated_qids if q in qrels}

        # Dense (uses cache if available)
        dense_results = run_dense_retrieval(
            corpus, queries,
            model_name=args.dense_model,
            top_k=args.top_k,
            cache_dir=args.dense_cache_dir,
            cache_key=f"{ds}::{args.split}::{args.dense_model}",
        )

        all_retrieval_data[ds] = {
            "bm25_results":  bm25_results,
            "dense_results": dense_results,
            "qrels":         qrels,
            "qrels_eval":    qrels_eval,
            "query_texts":   queries,
            "corpus":        corpus,
        }

        # Load best static alpha from existing run_summary (if available)
        summary_path = RESULTS_DIR / ds / "run_summary.json"
        if summary_path.exists():
            with open(summary_path, encoding="utf-8") as f:
                summary = json.load(f)
            best_static_by_dataset[ds] = summary.get("best_static_hybrid", {}).get("alpha", 0.5)
        else:
            best_static_by_dataset[ds] = 0.5

        print(f"  Loaded in {time.perf_counter() - t0:.1f}s  "
              f"(best_static_alpha={best_static_by_dataset[ds]})")

    # Run full Phase 4 evaluation
    results = run_full_evaluation(
        all_retrieval_data=all_retrieval_data,
        best_static_by_dataset=best_static_by_dataset,
        top_k=args.top_k,
        oracle_filename=args.oracle_filename,
        features_filename=args.features_filename,
        output_filename=output_filename,
        split_suffix=split_suffix,
    )

    print("\n" + "=" * 70)
    print(f"Phase 4 complete. Results saved to results/{output_filename}")


if __name__ == "__main__":
    main()
