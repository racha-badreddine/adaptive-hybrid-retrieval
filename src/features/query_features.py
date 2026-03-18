import re
import numpy as np
from typing import Dict, List


def simple_tokenize(text: str) -> List[str]:
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def compute_idf_dict(corpus: Dict) -> Dict[str, float]:
    """
    Compute a simple IDF dictionary from the corpus.
    """
    doc_count = len(corpus)
    term_doc_freq = {}

    for doc in corpus.values():
        title = doc.get("title", "") or ""
        text = doc.get("text", "") or ""
        content = f"{title} {text}".strip()

        tokens = set(simple_tokenize(content))
        for token in tokens:
            term_doc_freq[token] = term_doc_freq.get(token, 0) + 1

    idf = {}
    for token, df in term_doc_freq.items():
        idf[token] = np.log((doc_count + 1) / (df + 1)) + 1.0

    return idf


def get_top_score(results_for_query: Dict[str, float]) -> float:
    if not results_for_query:
        return 0.0
    return max(results_for_query.values())


def get_top_doc_id(results_for_query: Dict[str, float]) -> str:
    if not results_for_query:
        return ""
    return max(results_for_query.items(), key=lambda x: x[1])[0]


def compute_rank_of_doc(results_for_query: Dict[str, float], doc_id: str) -> int:
    """
    Return rank (1-based) of a doc_id in results. If not present, return large rank.
    """
    ranked = sorted(results_for_query.items(), key=lambda x: x[1], reverse=True)
    for rank, (d_id, _) in enumerate(ranked, start=1):
        if d_id == doc_id:
            return rank
    return len(ranked) + 1


def extract_query_features(
    query_text: str,
    bm25_results_for_query: Dict[str, float],
    dense_results_for_query: Dict[str, float],
    idf_dict: Dict[str, float],
) -> Dict[str, float]:
    """
    Extract query-level features for adaptive alpha(q).
    """
    tokens = simple_tokenize(query_text)

    query_length = len(tokens)

    token_idfs = [idf_dict.get(token, 0.0) for token in tokens]
    avg_idf = float(np.mean(token_idfs)) if token_idfs else 0.0
    idf_var = float(np.var(token_idfs)) if token_idfs else 0.0

    sparse_top1_score = get_top_score(bm25_results_for_query)
    dense_top1_score = get_top_score(dense_results_for_query)

    sparse_top1_doc = get_top_doc_id(bm25_results_for_query)
    dense_top1_doc = get_top_doc_id(dense_results_for_query)

    # disagreement feature:
    # rank of sparse top1 doc inside dense results + rank of dense top1 doc inside sparse results
    rank_sparse_in_dense = compute_rank_of_doc(dense_results_for_query, sparse_top1_doc)
    rank_dense_in_sparse = compute_rank_of_doc(bm25_results_for_query, dense_top1_doc)

    disagreement = float(rank_sparse_in_dense + rank_dense_in_sparse)

    return {
        "query_length": float(query_length),
        "avg_idf": avg_idf,
        "idf_var": idf_var,
        "sparse_top1_score": float(sparse_top1_score),
        "dense_top1_score": float(dense_top1_score),
        "disagreement": disagreement,
    }