import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from src.data.download_beir import download_beir_dataset
from src.data.load_beir import load_beir_dataset
from src.retrieval.bm25_retriever import run_bm25
from src.retrieval.dense_retriever import run_dense_retrieval
from src.retrieval.hybrid_static import fuse_static
from src.retrieval.rrf import reciprocal_rank_fusion
from beir.retrieval.evaluation import EvaluateRetrieval
from src.features.query_features import (
    FEATURE_NAMES,
    compute_idf_dict,
    extract_query_features,
)
from src.training.generate_oracle_alpha import generate_oracle_alphas, save_oracle_data
from src.training.analyze_oracle import analyze_oracle_data


# -- Metric helpers -------------------------------------------------------------

def _evaluate(qrels, results):
    """Return (ndcg, map, recall, precision) dicts from BEIR evaluator."""
    return EvaluateRetrieval.evaluate(qrels, results, [1, 3, 5, 10])


def _metrics_at_10(ndcg, _map, recall, precision):
    return {
        "NDCG@10":      ndcg["NDCG@10"],
        "Recall@10":    recall["Recall@10"],
        "MAP@10":       _map["MAP@10"],
        "Precision@10": precision["P@10"],
    }


def print_metrics(name: str, qrels, results) -> dict:
    m = _metrics_at_10(*_evaluate(qrels, results))
    print(f"\n{name}")
    print(f"  NDCG@10:      {m['NDCG@10']:.4f}")
    print(f"  Recall@10:    {m['Recall@10']:.4f}")
    print(f"  MAP@10:       {m['MAP@10']:.4f}")
    print(f"  Precision@10: {m['Precision@10']:.4f}")
    return m


# -- Save helpers ---------------------------------------------------------------

def save_run_summary(dataset_name: str, summary: dict) -> None:
    out_path = Path("results") / dataset_name / "run_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nRun summary saved -> {out_path}")


def save_features_data(
    dataset_name: str,
    all_features: dict,
    oracle_data: dict,
) -> None:
    """
    Combine extracted features with oracle alpha targets and save to
    results/{dataset}/features.json.
    """
    num_informative = sum(
        1 for v in oracle_data.values() if v.get("is_informative", False)
    )

    queries_out = {}
    for qid, feats in all_features.items():
        entry = oracle_data.get(qid, {})
        queries_out[qid] = {
            "features": feats,
            "oracle_alpha": entry.get("best_alpha"),
            "oracle_ndcg": entry.get("best_ndcg"),
            "is_informative": entry.get("is_informative", False),
        }

    output = {
        "metadata": {
            "dataset": dataset_name,
            "num_queries": len(queries_out),
            "num_informative": num_informative,
            "num_features": len(FEATURE_NAMES),
            "feature_names": FEATURE_NAMES,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        },
        "queries": queries_out,
    }

    out_path = Path("results") / dataset_name / "features.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Features saved        -> {out_path}  ({len(queries_out)} queries)")

    # Print brief feature statistics
    feature_matrix = np.array(
        [[feats[fn] for fn in FEATURE_NAMES] for feats in all_features.values()]
    )
    print("\nFeature statistics (mean +/- std):")
    for i, fn in enumerate(FEATURE_NAMES):
        col = feature_matrix[:, i]
        print(f"  {fn:<22} mean={col.mean():.4f}  std={col.std():.4f}"
              f"  min={col.min():.4f}  max={col.max():.4f}")


# -- Main pipeline --------------------------------------------------------------

def run_dataset_pipeline(
    dataset_name: str,
    data_dir: str,
    split: str = "test",
    top_k: int = 100,
    max_queries: int | None = None,
    dense_model: str = "all-MiniLM-L6-v2",
    dense_cache_dir: str | None = "results/cache/dense",
    run_oracle: bool = False,
):
    print(f"\n{'=' * 80}")
    print(f"Dataset: {dataset_name}")
    print(f"{'=' * 80}")

    download_beir_dataset(data_dir=data_dir, dataset_name=dataset_name)

    corpus, queries, qrels = load_beir_dataset(
        data_dir=data_dir,
        dataset_name=dataset_name,
        split=split,
    )

    print(f"Loaded {len(corpus):,} documents")
    print(f"Loaded {len(queries):,} queries")
    print(f"Loaded {len(qrels):,} qrels")

    # -- BM25 ------------------------------------------------------------------
    t0 = time.perf_counter()
    results_bm25 = run_bm25(corpus, queries, top_k=top_k, max_queries=max_queries)
    bm25_seconds = time.perf_counter() - t0

    # Build qrels for the evaluated query subset
    if max_queries is not None:
        evaluated_qids = set(results_bm25.keys())
        qrels_eval = {qid: qrels[qid] for qid in evaluated_qids if qid in qrels}
    else:
        qrels_eval = qrels

    bm25_metrics = print_metrics("BM25", qrels_eval, results_bm25)
    print(f"  Runtime: {bm25_seconds:.2f}s")

    # -- Dense -----------------------------------------------------------------
    t0 = time.perf_counter()
    results_dense = run_dense_retrieval(
        corpus,
        queries,
        model_name=dense_model,
        top_k=top_k,
        max_queries=max_queries,
        cache_dir=dense_cache_dir,
        cache_key=f"{dataset_name}::{split}::{dense_model}",
    )
    dense_seconds = time.perf_counter() - t0
    dense_metrics = print_metrics(f"Dense ({dense_model})", qrels_eval, results_dense)
    print(f"  Runtime: {dense_seconds:.2f}s")

    # -- RRF (k=60) ------------------------------------------------------------
    t0 = time.perf_counter()
    results_rrf = reciprocal_rank_fusion(
        bm25_results=results_bm25,
        dense_results=results_dense,
        k=60,
        top_k=top_k,
    )
    rrf_seconds = time.perf_counter() - t0
    rrf_metrics = print_metrics("RRF (k=60)", qrels_eval, results_rrf)
    print(f"  Runtime: {rrf_seconds:.2f}s")

    # -- Static Hybrid  alpha  in  {0.1, 0.2, ..., 0.9} --------------------------------
    alpha_values = [round(i / 10.0, 1) for i in range(1, 10)]  # 0.1 ... 0.9
    hybrid_results_by_alpha: dict = {}
    for alpha in alpha_values:
        t0 = time.perf_counter()
        results_hybrid = fuse_static(
            bm25_results=results_bm25,
            dense_results=results_dense,
            alpha=alpha,
            top_k=top_k,
        )
        hybrid_seconds = time.perf_counter() - t0
        m = print_metrics(f"Static Hybrid alpha={alpha:.1f}", qrels_eval, results_hybrid)
        print(f"  Runtime: {hybrid_seconds:.2f}s")
        hybrid_results_by_alpha[alpha] = {"metrics": m, "runtime_seconds": hybrid_seconds}

    # Best static hybrid by NDCG@10
    best_alpha = max(
        hybrid_results_by_alpha,
        key=lambda a: hybrid_results_by_alpha[a]["metrics"]["NDCG@10"],
    )
    best_static_metrics = hybrid_results_by_alpha[best_alpha]["metrics"]
    print(f"\n  Best static alpha = {best_alpha:.1f}  (NDCG@10={best_static_metrics['NDCG@10']:.4f})")

    # -- Build run_summary -----------------------------------------------------
    dense_key = "dense_" + dense_model.replace("/", "-").replace(".", "-").lower()

    run_summary: dict = {
        "metadata": {
            "dataset":              dataset_name,
            "split":                split,
            "top_k":                top_k,
            "max_queries":          max_queries,
            "num_docs":             len(corpus),
            "num_queries_total":    len(queries),
            "num_queries_evaluated": len(qrels_eval),
            "dense_model":          dense_model,
        },
        "bm25": {**bm25_metrics, "runtime_seconds": bm25_seconds},
        dense_key: {**dense_metrics, "runtime_seconds": dense_seconds},
        "rrf_k60": {**rrf_metrics, "runtime_seconds": rrf_seconds},
    }

    for alpha, info in hybrid_results_by_alpha.items():
        key = f"hybrid_alpha_{alpha:.1f}"
        run_summary[key] = {**info["metrics"], "runtime_seconds": info["runtime_seconds"]}

    run_summary["best_static_hybrid"] = {
        "alpha": best_alpha,
        **best_static_metrics,
    }

    # -- Oracle + Feature Extraction -------------------------------------------
    if run_oracle:
        print(f"\n{'-' * 60}")
        print("Oracle alpha generation (grid step=0.02) ...")
        # 51 values: 0.00, 0.02, ..., 1.00
        alpha_grid = [round(i / 50.0, 10) for i in range(51)]

        t0 = time.perf_counter()
        oracle_data = generate_oracle_alphas(
            qrels=qrels_eval,
            bm25_results=results_bm25,
            dense_results=results_dense,
            alpha_grid=alpha_grid,
            top_k=top_k,
        )
        oracle_seconds = time.perf_counter() - t0
        print(f"Oracle generation: {oracle_seconds:.1f}s")

        save_oracle_data(
            oracle_data,
            output_path=f"results/{dataset_name}/oracle_alpha.json",
        )

        oracle_stats = analyze_oracle_data(oracle_data)

        # Mean NDCG@10 achieved by the per-query oracle (upper bound)
        oracle_ndcg = float(np.mean([v["best_ndcg"] for v in oracle_data.values()]))

        run_summary["oracle"] = {
            "NDCG@10":       oracle_ndcg,
            "mean_alpha":    oracle_stats["mean_alpha"],
            "std_alpha":     oracle_stats["std_alpha"],
            "num_informative": oracle_stats["informative"],
            "num_total":     oracle_stats["total"],
            "runtime_seconds": oracle_seconds,
        }

        # -- Feature Extraction ------------------------------------------------
        print(f"\n{'-' * 60}")
        print("Extracting query features ...")
        idf_dict = compute_idf_dict(corpus)

        all_features: dict = {}
        skipped = 0
        for qid, query_text in queries.items():
            if qid not in results_bm25 or qid not in results_dense:
                skipped += 1
                continue
            all_features[qid] = extract_query_features(
                query_text=query_text,
                bm25_results_for_query=results_bm25[qid],
                dense_results_for_query=results_dense[qid],
                idf_dict=idf_dict,
            )

        print(f"Features extracted: {len(all_features)} queries  (skipped: {skipped})")
        save_features_data(dataset_name, all_features, oracle_data)

    save_run_summary(dataset_name, run_summary)
    return run_summary


# -- CLI ------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Run retrieval baselines on BEIR datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["scifact"],
        help="One or more BEIR dataset names (e.g., scifact arguana nfcorpus).",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory where BEIR datasets are stored/downloaded.",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (train/dev/test).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=100,
        help="Number of retrieved documents per query.",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="Optional cap on number of queries per dataset.",
    )
    parser.add_argument(
        "--dense-model",
        default="all-MiniLM-L6-v2",
        help="Sentence-transformers model for dense retrieval.",
    )
    parser.add_argument(
        "--dense-cache-dir",
        default="results/cache/dense",
        help="Directory for dense document embedding cache.",
    )
    parser.add_argument(
        "--disable-dense-cache",
        action="store_true",
        help="Disable loading/saving dense embedding cache.",
    )
    parser.add_argument(
        "--run-oracle",
        action="store_true",
        help="Run oracle alpha generation and full feature extraction (slower).",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    dense_cache_dir = None if args.disable_dense_cache else args.dense_cache_dir

    for dataset_name in args.datasets:
        run_dataset_pipeline(
            dataset_name=dataset_name,
            data_dir=args.data_dir,
            split=args.split,
            top_k=args.top_k,
            max_queries=args.max_queries,
            dense_model=args.dense_model,
            dense_cache_dir=dense_cache_dir,
            run_oracle=args.run_oracle,
        )


if __name__ == "__main__":
    main()
