"""
Post-retrieval feature extraction for adaptive hybrid retrieval.

12 new features organized into five groups:
  Group A - Normalized cross-method gap  (1): score_gap_norm
  Group B - Score margins                (3): margin_dense, margin_bm25, margin_ratio
  Group C - Rank overlap at wider K      (2): overlap_20, overlap_50
  Group D - Cross-scoring                (2): cross_dense_of_bm25top1, cross_bm25_of_densetop1
  Group E - Distributional shape         (4): var_bm25_top10, var_dense_top10,
                                              entropy_bm25_top10, entropy_dense_top10

Note: overlap@10 is omitted because it is linearly redundant with the existing
rank_disagreement feature (rank_disagreement = 1 - overlap@10).
"""

import numpy as np
from typing import Dict, List, Tuple


NEW_FEATURE_NAMES: List[str] = [
    # Group A
    "score_gap_norm",
    # Group B
    "margin_dense",
    "margin_bm25",
    "margin_ratio",
    # Group C
    "overlap_20",
    "overlap_50",
    # Group D
    "cross_dense_of_bm25top1",
    "cross_bm25_of_densetop1",
    # Group E
    "var_bm25_top10",
    "var_dense_top10",
    "entropy_bm25_top10",
    "entropy_dense_top10",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _min_max_norm(scores: Dict[str, float]) -> Dict[str, float]:
    """Per-query min-max normalization to [0, 1]. Constant scores -> 0.5."""
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-10:
        return {d: 0.5 for d in scores}
    return {d: (s - lo) / (hi - lo) for d, s in scores.items()}


def _softmax_entropy(arr: np.ndarray) -> float:
    """Shannon entropy of softmax-normalized scores. Zero for empty input."""
    if arr.size == 0:
        return 0.0
    shifted = arr - arr.max()
    exp_a = np.exp(shifted)
    probs = exp_a / (exp_a.sum() + 1e-10)
    return float(-np.sum(probs * np.log(probs + 1e-10)))


# ── Main extractor ────────────────────────────────────────────────────────────

def extract_post_retrieval_features(
    bm25_scores: Dict[str, float],
    dense_scores: Dict[str, float],
) -> Dict[str, float]:
    """
    Compute 12 post-retrieval features for a single query.

    Args:
        bm25_scores:  {doc_id: float}  BM25 raw scores for top-100 docs
        dense_scores: {doc_id: float}  Dense cosine scores for top-100 docs

    Returns:
        Dict with exactly the 12 keys listed in NEW_FEATURE_NAMES.
        Any missing data (empty dicts, < 2 results) is handled gracefully
        by returning 0.0 for the affected features.
    """
    # Sort each method's results descending by score
    bm25_ranked  = sorted(bm25_scores.items(),  key=lambda x: x[1], reverse=True)
    dense_ranked = sorted(dense_scores.items(), key=lambda x: x[1], reverse=True)

    bm25_docs,  bm25_vals  = ([d for d, _ in bm25_ranked],  [s for _, s in bm25_ranked])
    dense_docs, dense_vals = ([d for d, _ in dense_ranked], [s for _, s in dense_ranked])

    # ── A: Normalized score gap at top-1 ─────────────────────────────────────
    # Uses the same per-method min-max normalization as hybrid_static.py, then
    # subtracts: positive => dense top-1 normalizes higher, negative => BM25 does.
    bm25_norm  = _min_max_norm(bm25_scores)
    dense_norm = _min_max_norm(dense_scores)
    bm25_top1_norm  = bm25_norm.get(bm25_docs[0],   0.0) if bm25_docs  else 0.0
    dense_top1_norm = dense_norm.get(dense_docs[0],  0.0) if dense_docs else 0.0
    score_gap_norm  = dense_top1_norm - bm25_top1_norm

    # ── B: Score margins (retriever confidence) ───────────────────────────────
    margin_dense = float(dense_vals[0] - dense_vals[1]) if len(dense_vals) >= 2 else 0.0
    margin_bm25  = float(bm25_vals[0]  - bm25_vals[1])  if len(bm25_vals)  >= 2 else 0.0
    margin_ratio = margin_dense / (margin_dense + margin_bm25 + 1e-8)

    # ── C: Rank overlap at K=20 and K=50 ─────────────────────────────────────
    # K=10 is skipped: it is equivalent to (1 - rank_disagreement) which already exists.
    bm25_set20,  dense_set20  = set(bm25_docs[:20]),  set(dense_docs[:20])
    bm25_set50,  dense_set50  = set(bm25_docs[:50]),  set(dense_docs[:50])
    overlap_20 = len(bm25_set20  & dense_set20)  / 20.0
    overlap_50 = len(bm25_set50  & dense_set50)  / 50.0

    # ── D: Cross-scoring ──────────────────────────────────────────────────────
    # Does the other retriever also like this method's #1 document?
    bm25_top1_doc  = bm25_docs[0]  if bm25_docs  else None
    dense_top1_doc = dense_docs[0] if dense_docs else None
    cross_dense_of_bm25top1  = float(dense_scores.get(bm25_top1_doc,  0.0)) if bm25_top1_doc  else 0.0
    cross_bm25_of_densetop1  = float(bm25_scores.get(dense_top1_doc,  0.0)) if dense_top1_doc else 0.0

    # ── E: Distributional shape within top-10 ────────────────────────────────
    bm25_top10  = np.array(bm25_vals[:10],  dtype=float)
    dense_top10 = np.array(dense_vals[:10], dtype=float)

    var_bm25_top10      = float(np.var(bm25_top10))  if bm25_top10.size  > 0 else 0.0
    var_dense_top10     = float(np.var(dense_top10)) if dense_top10.size > 0 else 0.0
    entropy_bm25_top10  = _softmax_entropy(bm25_top10)
    entropy_dense_top10 = _softmax_entropy(dense_top10)

    return {
        "score_gap_norm":           float(score_gap_norm),
        "margin_dense":             float(margin_dense),
        "margin_bm25":              float(margin_bm25),
        "margin_ratio":             float(margin_ratio),
        "overlap_20":               float(overlap_20),
        "overlap_50":               float(overlap_50),
        "cross_dense_of_bm25top1":  cross_dense_of_bm25top1,
        "cross_bm25_of_densetop1":  cross_bm25_of_densetop1,
        "var_bm25_top10":           var_bm25_top10,
        "var_dense_top10":          var_dense_top10,
        "entropy_bm25_top10":       entropy_bm25_top10,
        "entropy_dense_top10":      entropy_dense_top10,
    }
