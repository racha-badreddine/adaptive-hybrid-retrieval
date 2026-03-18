from rank_bm25 import BM25Okapi
import re
from typing import Dict, Tuple


def simple_tokenize(text: str):
    """Very simple tokenizer for a first BM25 baseline."""
    text = text.lower()
    return re.findall(r"\b\w+\b", text)


def build_document_text(doc: Dict) -> str:
    """
    Combine title and text fields from a BEIR document.
    """
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    return f"{title} {text}".strip()


def run_bm25(corpus: Dict, queries: Dict, top_k: int = 100) -> Dict:
    """
    Run BM25 retrieval using rank_bm25 and return results in BEIR format.

    Args:
        corpus: BEIR corpus dict
        queries: BEIR queries dict
        top_k: number of documents to retrieve per query

    Returns:
        results: dict[query_id][doc_id] = score
    """
    doc_ids = list(corpus.keys())
    documents = [build_document_text(corpus[doc_id]) for doc_id in doc_ids]
    tokenized_corpus = [simple_tokenize(doc) for doc in documents]

    bm25 = BM25Okapi(tokenized_corpus)

    results = {}

    for query_id, query_text in queries.items():
        tokenized_query = simple_tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)

        ranked_indices = sorted(
            range(len(scores)),
            key=lambda i: scores[i],
            reverse=True
        )[:top_k]

        results[query_id] = {
            doc_ids[i]: float(scores[i])
            for i in ranked_indices
        }

    return results