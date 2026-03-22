import argparse
import json
import time
from pathlib import Path

from src.data.download_beir import download_beir_dataset
from src.data.load_beir import load_beir_dataset
from src.retrieval.bm25_retriever import run_bm25
from src.retrieval.dense_retriever import run_dense_retrieval
from src.retrieval.hybrid_static import fuse_static
from beir.retrieval.evaluation import EvaluateRetrieval
from src.features.query_features import compute_idf_dict, extract_query_features
from src.training.generate_oracle_alpha import generate_oracle_alphas, save_oracle_data
from src.training.analyze_oracle import analyze_oracle_data

def print_metrics(name, qrels, results):
    ndcg, _map, recall, precision = EvaluateRetrieval.evaluate(
        qrels, results, [1, 3, 5, 10]
    )

    print(f"\n{name} Results")
    print(f"NDCG@10: {ndcg['NDCG@10']:.4f}")
    print(f"Recall@10: {recall['Recall@10']:.4f}")
    print(f"MAP@10: {_map['MAP@10']:.4f}")
    print(f"P@10: {precision['P@10']:.4f}")

    return {
        "NDCG@10": ndcg["NDCG@10"],
        "Recall@10": recall["Recall@10"],
        "MAP@10": _map["MAP@10"],
        "P@10": precision["P@10"],
    }


def save_run_summary(dataset_name: str, summary: dict) -> None:
    out_path = Path("results") / dataset_name / "run_summary.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"Run summary saved to: {out_path}")


def run_dataset_pipeline(
    dataset_name: str,
    data_dir: str,
    split: str = "test",
    top_k: int = 100,
    max_queries: int | None = None,
    dense_model: str = "all-MiniLM-L6-v2",
    dense_cache_dir: str | None = "results/cache/dense",
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

    print(f"Loaded {len(corpus)} documents")
    print(f"Loaded {len(queries)} queries")
    print(f"Loaded {len(qrels)} qrels")

    t0 = time.perf_counter()
    results_bm25 = run_bm25(corpus, queries, top_k=top_k, max_queries=max_queries)
    bm25_seconds = time.perf_counter() - t0

    if max_queries is not None:
        evaluated_query_ids = set(results_bm25.keys())
        qrels_eval = {qid: qrels[qid] for qid in evaluated_query_ids if qid in qrels}
    else:
        qrels_eval = qrels

    bm25_metrics = print_metrics("BM25", qrels_eval, results_bm25)
    print(f"BM25 runtime: {bm25_seconds:.2f}s")

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
    dense_metrics = print_metrics("Dense", qrels_eval, results_dense)
    print(f"Dense runtime: {dense_seconds:.2f}s")

    hybrid_summaries = {}
    for alpha in [0.2, 0.5, 0.8]:
        t0 = time.perf_counter()
        results_hybrid = fuse_static(
            bm25_results=results_bm25,
            dense_results=results_dense,
            alpha=alpha,
            top_k=top_k,
        )
        hybrid_seconds = time.perf_counter() - t0
        hybrid_metrics = print_metrics(f"Static Hybrid (alpha={alpha})", qrels_eval, results_hybrid)
        print(f"Hybrid runtime (alpha={alpha}): {hybrid_seconds:.2f}s")
        hybrid_summaries[f"alpha_{alpha}"] = {
            "runtime_seconds": hybrid_seconds,
            "metrics": hybrid_metrics,
        }

    idf_dict = compute_idf_dict(corpus)

    sample_qid = list(queries.keys())[0]
    sample_features = extract_query_features(
        query_text=queries[sample_qid],
        bm25_results_for_query=results_bm25[sample_qid],
        dense_results_for_query=results_dense[sample_qid],
        idf_dict=idf_dict,
    )

    print("\nSample Query Features")
    print(f"Query ID: {sample_qid}")
    print(sample_features)
    
    
    oracle_data = generate_oracle_alphas(
        qrels=qrels_eval,
        bm25_results=results_bm25,
        dense_results=results_dense,
        alpha_grid=[i / 10.0 for i in range(11)],
        top_k=top_k,
    )

    save_oracle_data(
        oracle_data,
        output_path=f"results/{dataset_name}/oracle_alpha.json"
    )
    
    analyze_oracle_data(oracle_data)

    run_summary = {
        "dataset": dataset_name,
        "split": split,
        "top_k": top_k,
        "max_queries": max_queries,
        "num_docs": len(corpus),
        "num_queries_total": len(queries),
        "num_queries_evaluated": len(qrels_eval),
        "dense_model": dense_model,
        "dense_cache_dir": dense_cache_dir,
        "bm25": {
            "runtime_seconds": bm25_seconds,
            "metrics": bm25_metrics,
        },
        "dense": {
            "runtime_seconds": dense_seconds,
            "metrics": dense_metrics,
        },
        "hybrid": hybrid_summaries,
    }
    save_run_summary(dataset_name, run_summary)


def parse_args():
    parser = argparse.ArgumentParser(description="Run retrieval baselines on BEIR datasets.")
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
        help="Optional cap on number of queries per dataset for faster iteration.",
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
        )

if __name__ == "__main__":
    main()