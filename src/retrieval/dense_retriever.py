from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import random
from typing import Dict
from pathlib import Path
import hashlib

# BGE query prefix for retrieval tasks (documents use no prefix).
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# E5 prefixes: both queries and documents require a prefix.
E5_QUERY_PREFIX = "query: "
E5_DOC_PREFIX = "passage: "


def get_query_prefix(model_name: str) -> str:
    """Return the recommended query prefix for a given model name."""
    name = model_name.lower()
    if "bge" in name:
        return BGE_QUERY_PREFIX
    if "e5" in name:
        return E5_QUERY_PREFIX
    return ""


def get_doc_prefix(model_name: str) -> str:
    """Return the recommended document prefix for a given model name.
    Only E5 models require a document prefix; all others use no prefix.
    """
    if "e5" in model_name.lower():
        return E5_DOC_PREFIX
    return ""


def _make_cache_path(cache_dir: str, cache_key: str) -> Path:
    safe_key = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
    return Path(cache_dir) / f"{safe_key}.npz"


def _load_cached_embeddings(
    cache_path: Path,
    expected_doc_ids: list[str],
) -> np.ndarray | None:
    if not cache_path.exists():
        return None

    try:
        with np.load(cache_path) as data:
            cached_doc_ids = data["doc_ids"].tolist()
            cached_embeddings = data["embeddings"]

        if cached_doc_ids != expected_doc_ids:
            return None

        return cached_embeddings
    except Exception:
        return None


def _save_cached_embeddings(cache_path: Path, doc_ids: list[str], embeddings: np.ndarray) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        doc_ids=np.array(doc_ids),
        embeddings=embeddings,
    )


def build_document_text(doc: Dict) -> str:
    """
    Combine title and text fields from a BEIR document.
    """
    title = doc.get("title", "") or ""
    text = doc.get("text", "") or ""
    return f"{title} {text}".strip()


def run_dense_retrieval(
    corpus: Dict,
    queries: Dict,
    model_name: str = "all-MiniLM-L6-v2",
    top_k: int = 100,
    max_queries: int | None = None,
    cache_dir: str | None = None,
    cache_key: str | None = None,
    query_prefix: str | None = None,
    doc_prefix: str | None = None,
):
    """
    Run dense retrieval using Sentence-BERT embeddings.

    Args:
        corpus: BEIR corpus dictionary
        queries: BEIR queries dictionary
        model_name: sentence-transformer model name
        top_k: number of documents to retrieve per query
        max_queries: optional limit on number of queries to process
        cache_dir: optional directory for storing dense doc embedding cache
        cache_key: optional cache identity (e.g., dataset + model)
        query_prefix: prefix prepended to query texts before encoding.
            None = auto-detect from model_name; "" = force no prefix.
        doc_prefix: prefix prepended to document texts before encoding.
            None = auto-detect from model_name; "" = force no prefix.
            E5 models require "passage: "; all others use no prefix.

    Returns:
        results: dict[query_id][doc_id] = score
    """
    model = SentenceTransformer(model_name)

    # Auto-detect prefixes when not explicitly overridden
    if query_prefix is None:
        query_prefix = get_query_prefix(model_name)
    if doc_prefix is None:
        doc_prefix = get_doc_prefix(model_name)

    doc_ids = list(corpus.keys())
    raw_documents = [build_document_text(corpus[doc_id]) for doc_id in doc_ids]

    # Apply document prefix before encoding (and caching)
    if doc_prefix:
        documents = [f"{doc_prefix}{d}" for d in raw_documents]
    else:
        documents = raw_documents

    query_ids = list(queries.keys())
    if max_queries is not None:
        # Use a fixed seed to ensure BM25 and Dense retrievers select the exact same queries
        rng = random.Random(42)
        query_ids = rng.sample(query_ids, min(max_queries, len(query_ids)))

    query_texts = [queries[qid] for qid in query_ids]

    doc_embeddings = None
    cache_path = None
    if cache_dir and cache_key:
        cache_path = _make_cache_path(cache_dir, cache_key)
        doc_embeddings = _load_cached_embeddings(cache_path, doc_ids)
        if doc_embeddings is not None:
            print(f"Loaded dense doc embeddings from cache: {cache_path}")

    if doc_embeddings is None:
        doc_embeddings = model.encode(
            documents,
            batch_size=32,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        if cache_path is not None:
            _save_cached_embeddings(cache_path, doc_ids, doc_embeddings)
            print(f"Saved dense doc embeddings to cache: {cache_path}")

    # Apply query prefix only to query texts (not documents)
    if query_prefix:
        query_texts_to_encode = [f"{query_prefix}{t}" for t in query_texts]
    else:
        query_texts_to_encode = query_texts

    query_embeddings = model.encode(
        query_texts_to_encode,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    # Cosine similarity
    sim_matrix = cosine_similarity(query_embeddings, doc_embeddings)

    results = {}

    for i, query_id in enumerate(query_ids):
        scores = sim_matrix[i]
        ranked_indices = np.argsort(scores)[::-1][:top_k]

        results[query_id] = {
            doc_ids[idx]: float(scores[idx])
            for idx in ranked_indices
        }

    return results



#Later, if needed:

#cache embeddings

#use FAISS

#avoid recomputing docs