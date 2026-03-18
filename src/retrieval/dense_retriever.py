from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from typing import Dict


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
):
    """
    Run dense retrieval using Sentence-BERT embeddings.

    Args:
        corpus: BEIR corpus dictionary
        queries: BEIR queries dictionary
        model_name: sentence-transformer model name
        top_k: number of documents to retrieve per query

    Returns:
        results: dict[query_id][doc_id] = score
    """
    model = SentenceTransformer(model_name)

    doc_ids = list(corpus.keys())
    documents = [build_document_text(corpus[doc_id]) for doc_id in doc_ids]

    query_ids = list(queries.keys())
    query_texts = [queries[qid] for qid in query_ids]

    # Encode documents and queries
    doc_embeddings = model.encode(
        documents,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    query_embeddings = model.encode(
        query_texts,
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