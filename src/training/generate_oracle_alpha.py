from typing import Dict, List, Tuple
from beir.retrieval.evaluation import EvaluateRetrieval
import json
from pathlib import Path
from src.retrieval.hybrid_static import fuse_single_query


def evaluate_single_query_ndcg(
    query_id: str,
    qrels: Dict[str, Dict[str, int]],
    fused_results_for_query: Dict[str, float],
    k_values: List[int] = [10],
) -> float:
    """
    Evaluate a single query using NDCG@10.
    """
    qrels_single = {query_id: qrels[query_id]}
    results_single = {query_id: fused_results_for_query}

    ndcg, _, _, _ = EvaluateRetrieval.evaluate(
        qrels_single,
        results_single,
        k_values,
    )
    return ndcg["NDCG@10"]


def generate_oracle_alphas(
    qrels: Dict[str, Dict[str, int]],
    bm25_results: Dict[str, Dict[str, float]],
    dense_results: Dict[str, Dict[str, float]],
    alpha_grid: List[float] = None,
    top_k: int = 100,
) -> Dict[str, Dict]:
    """
    For each query, find the best alpha from a grid search.

    Returns:
        oracle_data[query_id] = {
            "best_alpha": ...,
            "best_ndcg": ...,
            "alpha_scores": {alpha: ndcg}
        }
    """
    if alpha_grid is None:
        alpha_grid = [i / 10.0 for i in range(11)]  # 0.0 to 1.0

    oracle_data = {}

    for query_id in qrels.keys():
        bm25_q = bm25_results.get(query_id, {})
        dense_q = dense_results.get(query_id, {})

        alpha_scores = {}

        for alpha in alpha_grid:
            fused_q = fuse_single_query(
                bm25_results_for_query=bm25_q,
                dense_results_for_query=dense_q,
                alpha=alpha,
                top_k=top_k,
            )

            ndcg = evaluate_single_query_ndcg(
                query_id=query_id,
                qrels=qrels,
                fused_results_for_query=fused_q,
                k_values=[10],
            )

            alpha_scores[alpha] = ndcg

        best_ndcg = max(alpha_scores.values())

        best_alphas = [
            alpha for alpha, score in alpha_scores.items()
            if score == best_ndcg
        ]

        # choose the middle alpha among ties instead of always the first
        best_alpha = best_alphas[len(best_alphas) // 2]

        score_values = list(alpha_scores.values())
        is_informative = len(set(score_values)) > 1 and best_ndcg > 0.0

        oracle_data[query_id] = {
            "best_alpha": best_alpha,
            "best_ndcg": best_ndcg,
            "alpha_scores": alpha_scores,
            "is_informative": is_informative,
            "num_best_alphas": len(best_alphas),
        }

    return oracle_data

def save_oracle_data(oracle_data: Dict, output_path: str):
    """
    Save oracle alpha data to JSON file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(oracle_data, f, indent=2)

    print(f"Oracle data saved to: {output_path}")