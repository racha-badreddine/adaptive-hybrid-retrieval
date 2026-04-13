"""
Query feature extraction for adaptive hybrid retrieval.

17 features organized into four groups:
  Group 1 – Query-Intrinsic  (5): query_length, avg_idf, idf_variance, max_idf, query_entropy
  Group 2 – Sparse-Signal    (4): bm25_top1_score, bm25_top10_mean, bm25_top10_std, bm25_score_gap
  Group 3 – Dense-Signal     (4): dense_top1_score, dense_top10_mean, dense_top10_std, dense_score_gap
  Group 4 – Cross-Retriever  (4): rank_disagreement, score_ratio, kendall_tau, rbo_score
"""

import re
import numpy as np
from typing import Dict, List, Tuple
from scipy.stats import kendalltau as scipy_kendalltau


FEATURE_NAMES: List[str] = [
    # Group 1: Query-Intrinsic
    "query_length",
    "avg_idf",
    "idf_variance",
    "max_idf",
    "query_entropy",
    # Group 2: Sparse-Signal
    "bm25_top1_score",
    "bm25_top10_mean",
    "bm25_top10_std",
    "bm25_score_gap",
    # Group 3: Dense-Signal
    "dense_top1_score",
    "dense_top10_mean",
    "dense_top10_std",
    "dense_score_gap",
    # Group 4: Cross-Retriever
    "rank_disagreement",
    "score_ratio",
    "kendall_tau",
    "rbo_score",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def compute_idf_dict(corpus: Dict) -> Dict[str, float]:
    """
    Compute corpus-level IDF for every token.
    Formula: log((N+1)/(df+1)) + 1.0  (smoothed, consistent with sklearn)
    """
    doc_count = len(corpus)
    term_doc_freq: Dict[str, int] = {}

    for doc in corpus.values():
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        content = f"{title} {text}".strip()
        tokens = set(simple_tokenize(content))
        for token in tokens:
            term_doc_freq[token] = term_doc_freq.get(token, 0) + 1

    idf: Dict[str, float] = {}
    for token, df in term_doc_freq.items():
        idf[token] = np.log((doc_count + 1) / (df + 1)) + 1.0

    return idf


def _top_k_sorted(results: Dict[str, float], k: int = 10) -> List[Tuple[str, float]]:
    """Return up to k (doc_id, score) pairs sorted by score descending."""
    return sorted(results.items(), key=lambda x: x[1], reverse=True)[:k]


def _min_max_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    """Per-query min-max normalization into [0, 1]. Ties → 0.5."""
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi == lo:
        return {doc_id: 0.5 for doc_id in scores}
    return {doc_id: (s - lo) / (hi - lo) for doc_id, s in scores.items()}


def _compute_rbo(list1: List[str], list2: List[str], p: float = 0.9) -> float:
    """
    Rank-Biased Overlap (Webber et al., 2010) with persistence p.
    RBO = (1-p) * Σ_{d=1}^{k}  p^{d-1} * |S[:d] ∩ T[:d]| / d
    """
    k = min(len(list1), len(list2))
    if k == 0:
        return 0.0
    rbo_sum = 0.0
    for d in range(1, k + 1):
        overlap = len(set(list1[:d]) & set(list2[:d]))
        rbo_sum += (p ** (d - 1)) * (overlap / d)
    return (1.0 - p) * rbo_sum


# ── Main feature extractor ────────────────────────────────────────────────────

def extract_query_features(
    query_text: str,
    bm25_results_for_query: Dict[str, float],
    dense_results_for_query: Dict[str, float],
    idf_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Extract all 17 query-level features for a single query.

    Args:
        query_text:              Raw query string.
        bm25_results_for_query:  {doc_id: bm25_score} for this query.
        dense_results_for_query: {doc_id: dense_score} for this query.
        idf_dict:                Corpus-level IDF mapping (from compute_idf_dict).

    Returns:
        Dictionary with exactly the 17 keys listed in FEATURE_NAMES.
    """

    # ── Group 1: Query-Intrinsic ──────────────────────────────────────────────
    tokens = simple_tokenize(query_text)
    query_length = len(tokens)
    token_idfs = [idf_dict.get(t, 0.0) for t in tokens]

    avg_idf = float(np.mean(token_idfs)) if token_idfs else 0.0
    idf_variance = float(np.var(token_idfs)) if token_idfs else 0.0
    max_idf = float(max(token_idfs)) if token_idfs else 0.0

    # Shannon entropy of IDF values treated as a probability distribution.
    # If all IDFs are 0 or there is only one unique token, entropy = 0.
    if token_idfs and sum(token_idfs) > 0.0:
        arr = np.array(token_idfs, dtype=float)
        arr = arr / arr.sum()          # normalize to sum-to-1
        arr = arr[arr > 0]             # avoid log(0)
        query_entropy = float(-np.sum(arr * np.log2(arr)))
    else:
        query_entropy = 0.0

    # ── Group 2: Sparse-Signal ────────────────────────────────────────────────
    bm25_top10 = _top_k_sorted(bm25_results_for_query, k=10)
    bm25_top10_scores = [s for _, s in bm25_top10]

    bm25_top1_score = bm25_top10_scores[0] if bm25_top10_scores else 0.0
    bm25_top10_mean = float(np.mean(bm25_top10_scores)) if bm25_top10_scores else 0.0
    bm25_top10_std = float(np.std(bm25_top10_scores)) if bm25_top10_scores else 0.0
    bm25_score_gap = (
        bm25_top10_scores[0] - bm25_top10_scores[-1]
        if len(bm25_top10_scores) >= 2
        else 0.0
    )

    # ── Group 3: Dense-Signal ─────────────────────────────────────────────────
    dense_top10 = _top_k_sorted(dense_results_for_query, k=10)
    dense_top10_scores = [s for _, s in dense_top10]

    dense_top1_score = dense_top10_scores[0] if dense_top10_scores else 0.0
    dense_top10_mean = float(np.mean(dense_top10_scores)) if dense_top10_scores else 0.0
    dense_top10_std = float(np.std(dense_top10_scores)) if dense_top10_scores else 0.0
    dense_score_gap = (
        dense_top10_scores[0] - dense_top10_scores[-1]
        if len(dense_top10_scores) >= 2
        else 0.0
    )

    # ── Group 4: Cross-Retriever ──────────────────────────────────────────────
    bm25_top10_ids = [doc_id for doc_id, _ in bm25_top10]
    dense_top10_ids = [doc_id for doc_id, _ in dense_top10]
    bm25_id_set = set(bm25_top10_ids)
    dense_id_set = set(dense_top10_ids)

    # rank_disagreement: fraction of top-10 docs not shared.
    # Use 10 as denominator only when both lists have ≥ 10 results.
    denom = min(len(bm25_top10_ids), len(dense_top10_ids), 10)
    if denom > 0:
        overlap = len(bm25_id_set & dense_id_set)
        rank_disagreement = 1.0 - (overlap / denom)
    else:
        rank_disagreement = 0.0

    # score_ratio: raw BM25 top-1 score divided by raw dense top-1 score.
    # Captures relative score magnitude between retrievers. StandardScaler
    # handles the scale difference at training time.
    score_ratio = float(bm25_top1_score) / max(float(dense_top1_score), 1e-8)

    # kendall_tau: rank correlation on documents shared between both top-10 lists.
    shared = list(bm25_id_set & dense_id_set)
    if len(shared) >= 2:
        bm25_ranks = [bm25_top10_ids.index(d) for d in shared]
        dense_ranks = [dense_top10_ids.index(d) for d in shared]
        tau, _ = scipy_kendalltau(bm25_ranks, dense_ranks)
        kendall_tau_val = float(tau) if not np.isnan(tau) else 0.0
    else:
        kendall_tau_val = 0.0

    # rbo_score: Rank-Biased Overlap between the two top-10 ranked lists.
    rbo_score = _compute_rbo(bm25_top10_ids, dense_top10_ids, p=0.9)

    return {
        # Group 1
        "query_length":   float(query_length),
        "avg_idf":        avg_idf,
        "idf_variance":   idf_variance,
        "max_idf":        max_idf,
        "query_entropy":  query_entropy,
        # Group 2
        "bm25_top1_score":  float(bm25_top1_score),
        "bm25_top10_mean":  bm25_top10_mean,
        "bm25_top10_std":   bm25_top10_std,
        "bm25_score_gap":   bm25_score_gap,
        # Group 3
        "dense_top1_score": float(dense_top1_score),
        "dense_top10_mean": dense_top10_mean,
        "dense_top10_std":  dense_top10_std,
        "dense_score_gap":  dense_score_gap,
        # Group 4
        "rank_disagreement": rank_disagreement,
        "score_ratio":       score_ratio,
        "kendall_tau":       kendall_tau_val,
        "rbo_score":         float(rbo_score),
    }
