from rank_bm25 import BM25Okapi
import re
import random
from typing import Dict, Tuple
import numpy as np


# THis later needs to be replaced with a more robust tokenizer, but this is a simple start for BM25 baseline.
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


def run_bm25(
    corpus: Dict,
    queries: Dict,
    top_k: int = 100,
    max_queries: int | None = None,
) -> Dict:
    """
    Run BM25 retrieval using rank_bm25 and return results in BEIR format.

    Args:
        corpus: BEIR corpus dict
        queries: BEIR queries dict
        top_k: number of documents to retrieve per query
        max_queries: optional limit on number of queries to process

    Returns:
        results: dict[query_id][doc_id] = score
    """
    doc_ids = list(corpus.keys())
    documents = [build_document_text(corpus[doc_id]) for doc_id in doc_ids]
    tokenized_corpus = [simple_tokenize(doc) for doc in documents]

    bm25 = BM25Okapi(tokenized_corpus)

    results = {}

    query_items = list(queries.items())
    if max_queries is not None:
        # Use a fixed seed to ensure BM25 and Dense retrievers select the exact same queries
        rng = random.Random(42)
        query_items = rng.sample(query_items, min(max_queries, len(query_items)))

    for query_id, query_text in query_items:
        tokenized_query = simple_tokenize(query_text)
        scores = bm25.get_scores(tokenized_query)

        if top_k >= len(scores):
            ranked_indices = np.argsort(scores)[::-1]
        else:
            # Faster partial top-k selection than full sort on large corpora.
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            ranked_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results[query_id] = {
            doc_ids[int(i)]: float(scores[int(i)])
            for i in ranked_indices
        }

    return results