"""
Reciprocal Rank Fusion (RRF) baseline.

Reference: Cormack et al. (2009) "Reciprocal Rank Fusion outperforms Condorcet
and individual Rank Learning Methods".

Formula (per document d, across retrievers R):
    RRF_score(d) = Σ_{r ∈ R}  1 / (k + rank_r(d))

Documents absent from a ranking are treated as having rank = len(ranking) + 1
(i.e. they contribute a small but non-zero score).
"""

from typing import Dict


def reciprocal_rank_fusion(
    bm25_results: Dict[str, Dict[str, float]],
    dense_results: Dict[str, Dict[str, float]],
    k: int = 60,
    top_k: int = 100,
) -> Dict[str, Dict[str, float]]:
    """
    Apply RRF to all queries.

    Args:
        bm25_results:   {query_id: {doc_id: score}} from BM25.
        dense_results:  {query_id: {doc_id: score}} from dense retrieval.
        k:              RRF smoothing constant (default 60, standard in literature).
        top_k:          Number of documents to keep per query.

    Returns:
        {query_id: {doc_id: rrf_score}} for all queries in bm25_results.
    """
    fused: Dict[str, Dict[str, float]] = {}
    for qid in bm25_results:
        fused[qid] = _rrf_single_query(
            bm25_results_for_query=bm25_results[qid],
            dense_results_for_query=dense_results.get(qid, {}),
            k=k,
            top_k=top_k,
        )
    return fused


def _rrf_single_query(
    bm25_results_for_query: Dict[str, float],
    dense_results_for_query: Dict[str, float],
    k: int = 60,
    top_k: int = 100,
) -> Dict[str, float]:
    """
    RRF for a single query.
    Absent documents receive rank = len(their_list) + 1 (contributes small score).
    """
    # Build rank maps (1-based, sorted by score descending)
    bm25_ranked = sorted(bm25_results_for_query.items(), key=lambda x: x[1], reverse=True)
    dense_ranked = sorted(dense_results_for_query.items(), key=lambda x: x[1], reverse=True)

    bm25_rank: Dict[str, int] = {doc_id: r + 1 for r, (doc_id, _) in enumerate(bm25_ranked)}
    dense_rank: Dict[str, int] = {doc_id: r + 1 for r, (doc_id, _) in enumerate(dense_ranked)}

    # Fallback ranks for absent documents
    bm25_absent_rank = len(bm25_ranked) + 1
    dense_absent_rank = len(dense_ranked) + 1

    all_docs = set(bm25_rank) | set(dense_rank)

    rrf_scores: Dict[str, float] = {}
    for doc_id in all_docs:
        r_bm25 = bm25_rank.get(doc_id, bm25_absent_rank)
        r_dense = dense_rank.get(doc_id, dense_absent_rank)
        rrf_scores[doc_id] = 1.0 / (k + r_bm25) + 1.0 / (k + r_dense)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return dict(ranked)
