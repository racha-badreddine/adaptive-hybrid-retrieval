from typing import Dict, Set


def min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """
    Min-max normalize a score dictionary for one query.
    """
    if not scores:
        return {}

    values = list(scores.values())
    min_score = min(values)
    max_score = max(values)

    if max_score == min_score:
        return {doc_id: 0.0 for doc_id in scores}

    return {
        doc_id: (score - min_score) / (max_score - min_score)
        for doc_id, score in scores.items()
    }


def fuse_static(
    bm25_results: Dict[str, Dict[str, float]],
    dense_results: Dict[str, Dict[str, float]],
    alpha: float = 0.5,
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Fuse BM25 and dense retrieval results using a fixed alpha.

    score = alpha * sparse + (1 - alpha) * dense

    Args:
        bm25_results: dict[query_id][doc_id] = bm25 score
        dense_results: dict[query_id][doc_id] = dense score
        alpha: weight for BM25
        top_k: number of documents to keep per query

    Returns:
        fused_results: dict[query_id][doc_id] = fused score
    """
    fused_results = {}

    query_ids = bm25_results.keys()

    for qid in query_ids:
        bm25_q = min_max_normalize(bm25_results.get(qid, {}))
        dense_q = min_max_normalize(dense_results.get(qid, {}))

        all_doc_ids: Set[str] = set(bm25_q.keys()) | set(dense_q.keys())

        fused_scores = {}
        for doc_id in all_doc_ids:
            sparse_score = bm25_q.get(doc_id, 0.0)
            dense_score = dense_q.get(doc_id, 0.0)
            fused_scores[doc_id] = alpha * sparse_score + (1.0 - alpha) * dense_score

        ranked = sorted(
            fused_scores.items(),
            key=lambda x: x[1],
            reverse=True
        )[:top_k]

        fused_results[qid] = dict(ranked)

    return fused_results

def fuse_single_query(
    bm25_results_for_query: Dict[str, float],
    dense_results_for_query: Dict[str, float],
    alpha: float = 0.5,
    top_k: int = 100,
) -> Dict[str, float]:
    """
    Fuse BM25 and dense scores for a single query.
    """
    bm25_q = min_max_normalize(bm25_results_for_query)
    dense_q = min_max_normalize(dense_results_for_query)

    all_doc_ids = set(bm25_q.keys()) | set(dense_q.keys())

    fused_scores = {}
    for doc_id in all_doc_ids:
        sparse_score = bm25_q.get(doc_id, 0.0)
        dense_score = dense_q.get(doc_id, 0.0)
        fused_scores[doc_id] = alpha * sparse_score + (1.0 - alpha) * dense_score

    ranked = sorted(
        fused_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_k]

    return dict(ranked)